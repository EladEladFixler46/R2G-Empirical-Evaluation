from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from r2g_eval.algorithms import BaseR2GAlgorithm
from r2g_eval.models import GraphInstance, ProblemInstance, RDBInstance


from torch_geometric.utils import add_self_loops

import os
import shutil
import tempfile
import uuid
from torch.utils.data import Dataset

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")          
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.activation = nn.PReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(torch.cat([x_i, x_j], dim=-1)))

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:          
        return self.norm(aggr_out)


class MPNN(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 32, num_layers: int = 3):
        super().__init__()
        layers = []
        current_in = in_channels
        for _ in range(num_layers):
            layers.append(MPNNLayer(current_in, hidden_dim))
            current_in = hidden_dim
            
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.head(x).squeeze(-1)   


def _graph_instance_to_pyg(gi: GraphInstance) -> Data:
    return Data(
        x=gi.embeddings.float(),
        edge_index=gi.edge_index,
        num_nodes=gi.embeddings.shape[0],
    )


def _problem_to_labelled_pyg(
    pi: ProblemInstance,
    algorithm: BaseR2GAlgorithm,
) -> tuple[Data, dict[int, float]]:
    gi = algorithm._run(pi.rdb_instance)
    data = _graph_instance_to_pyg(gi)

    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    y = torch.full((data.num_nodes,), float("nan"))
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    id_to_node: dict[str, int] = {v: k for k, v in gi.node_to_id.items()}

    if pi.expected_properties:
        keys = np.array(list(pi.expected_properties.keys()))
        vals = np.array(list(pi.expected_properties.values()))
        
        valid_mask = np.isin(keys, list(id_to_node.keys()))
        valid_keys = keys[valid_mask]
        valid_vals = vals[valid_mask]
        
        if len(valid_keys) > 0:
            indices = [id_to_node[k] for k in valid_keys]
            y[indices] = torch.tensor(valid_vals, dtype=torch.float)
            mask[indices] = True

    data.y = y
    data.labelled_mask = mask
    return data, id_to_node


@dataclass
class TrainedMPNNModel:
    mpnn: MPNN
    algorithm: BaseR2GAlgorithm
    device: torch.device

    def predict(self, rdb_instance: RDBInstance) -> dict[str, float]:
        gi = self.algorithm._run(rdb_instance)
        data = _graph_instance_to_pyg(gi)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        data = data.to(self.device)

        self.mpnn.eval()
        with torch.no_grad():
            logits = self.mpnn(data.x, data.edge_index)   
            preds = torch.sigmoid(logits)

        return {
            gi.node_to_id[node_idx]: preds[node_idx].item()
            for node_idx in gi.node_to_id
        }


class RAMGraphDataset(Dataset):
    def __init__(self, problems: list[ProblemInstance], algorithm: BaseR2GAlgorithm, split_name: str = "train"):
        print(f"Pre-computing {split_name} graphs into RAM ({len(problems)} instances)...")
        self._cache: list[Data] = []
        for p in problems:
            data, _ = _problem_to_labelled_pyg(p, algorithm)
            self._cache.append(data)
        print(f"  Done - {len(self._cache)} graphs cached.")

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Data:
        return self._cache[idx]

    def cleanup(self) -> None:
        pass


def _get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class TrainingConfig:
    hidden_dim:  int   = 32
    num_layers:  int   = 3
    lr:          float = 1e-2
    epochs:      int   = 50
    batch_size:  int   = 32     
    device:      str   = field(default_factory=_get_default_device)


def train(
    train_problems: list[ProblemInstance],
    test_problems:  list[ProblemInstance],
    algorithm:      BaseR2GAlgorithm,
    config:         TrainingConfig | None = None,
    loss_fn:        nn.Module | None = None,
) -> tuple[TrainedMPNNModel, dict[str, Any]]:
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)
    print(f"Training on device: {device}")

    train_data = RAMGraphDataset(train_problems, algorithm, split_name="train")
    test_data  = RAMGraphDataset(test_problems,  algorithm, split_name="test")

    from torch_geometric.loader import DataLoader
    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_data,  batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    first_train_graph = train_data[0]
    in_channels = 1 if not hasattr(first_train_graph, 'x') or first_train_graph.x.shape[1] == 0 else first_train_graph.x.shape[1]

    mpnn = MPNN(
        in_channels=in_channels, 
        hidden_dim=config.hidden_dim, 
        num_layers=config.num_layers
    ).to(device)
    
    optimiser = torch.optim.Adam(mpnn.parameters(), lr=config.lr)
    
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = loss_fn.to(device)

    history: dict[str, list[float]] = {"train_loss": [], "test_loss": []}

    for epoch in range(1, config.epochs + 1):
        mpnn.train()
        total_train_loss = 0.0
        n_train_batches  = 0

        for data in train_loader:
            data = data.to(device)
            if data.labelled_mask.sum() == 0:
                continue

            optimiser.zero_grad()
            logits  = mpnn(data.x, data.edge_index)
            y_true = data.y[data.labelled_mask]
            y_pred = logits[data.labelled_mask]
            loss   = loss_fn(y_pred, y_true)
            loss.backward()
            optimiser.step()

            total_train_loss += loss.item()
            n_train_batches  += 1

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        mpnn.eval()
        total_test_loss = 0.0
        n_test_batches  = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                if data.labelled_mask.sum() == 0:
                    continue
                logits  = mpnn(data.x, data.edge_index)
                y_true = data.y[data.labelled_mask]
                y_pred = logits[data.labelled_mask]
                loss   = loss_fn(y_pred, y_true)
                
                total_test_loss += loss.item()
                n_test_batches  += 1

        avg_test_loss = total_test_loss / max(n_test_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{config.epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"test_loss={avg_test_loss:.4f}"
            )

    trained_model = TrainedMPNNModel(mpnn=mpnn, algorithm=algorithm, device=device)
    return trained_model, history


@dataclass
class EvaluationResults:
    accuracy:          float
    precision:         float
    recall:            float
    f1:                float
    auc_roc:           float
    avg_loss:          float
    threshold:         float
    per_instance:      list[dict[str, Any]] = field(default_factory=list)
    y_true:            list[float] = field(default_factory=list)
    y_prob:            list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "==============================",
            "      Evaluation Results      ",
            "==============================",
            f"  Accuracy   : {self.accuracy:.4f}",
            f"  Precision  : {self.precision:.4f}",
            f"  Recall     : {self.recall:.4f}",
            f"  F1 Score   : {self.f1:.4f}",
            f"  AUC-ROC    : {self.auc_roc:.4f}",
            f"  Avg Loss   : {self.avg_loss:.4f}",
            f"  Threshold  : {self.threshold:.2f}",
            "==============================",
        ]
        return "\n".join(lines)


def evaluate(
    model:    TrainedMPNNModel,
    problems: list[ProblemInstance],
    threshold: float = 0.5,
    loss_fn:   nn.Module | None = None,
) -> EvaluationResults:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )

    device   = model.device
    
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = loss_fn.to(device)

    all_true:  list[float] = []
    all_pred:  list[float] = []
    per_instance: list[dict[str, Any]] = []

    model.mpnn.eval()
    with torch.no_grad():
        for pi in problems:
            gi   = model.algorithm._run(pi.rdb_instance)
            data = _graph_instance_to_pyg(gi)
            data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
            data = data.to(device)

            logits: torch.Tensor = model.mpnn(data.x, data.edge_index)
            preds: torch.Tensor = torch.sigmoid(logits)

            id_to_node: dict[str, int] = {v: k for k, v in gi.node_to_id.items()}

            if not pi.expected_properties:
                continue

            keys = np.array(list(pi.expected_properties.keys()))
            vals = np.array(list(pi.expected_properties.values()))
            
            valid_mask = np.isin(keys, list(id_to_node.keys()))
            valid_keys = keys[valid_mask]
            valid_vals = vals[valid_mask]

            if len(valid_keys) == 0:
                continue
                
            indices = [id_to_node[k] for k in valid_keys]
            inst_true = valid_vals.tolist()
            inst_logits = logits[indices]
            inst_probs = preds[indices].tolist()

            t_arr = torch.tensor(inst_true, dtype=torch.float, device=device)
            inst_loss = loss_fn(inst_logits, t_arr).item()

            bin_pred = [1 if p >= threshold else 0 for p in inst_probs]

            per_instance.append({
                "instance_id":  pi.instance_id,
                "loss":         inst_loss,
                "n_nodes":      len(inst_true),
                "n_positive":   int(sum(inst_true)),
                "n_predicted_positive": int(sum(bin_pred)),
            })

            all_true.extend(inst_true)
            all_pred.extend(inst_probs)

    if not all_true:
        return EvaluationResults(
            accuracy=0, precision=0, recall=0, f1=0,
            auc_roc=0, avg_loss=0, threshold=threshold,
            per_instance=per_instance,
            y_true=[], y_prob=[]
        )

    y_true = np.array(all_true)
    y_prob = np.array(all_pred)
    y_bin  = (y_prob >= threshold).astype(int)

    bce_loss = nn.BCELoss()
    avg_loss  = bce_loss(torch.tensor(y_prob, dtype=torch.float), torch.tensor(y_true, dtype=torch.float)).item()
    accuracy  = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall    = recall_score(y_true, y_bin, zero_division=0)
    f1        = f1_score(y_true, y_bin, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = float("nan")

    return EvaluationResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        avg_loss=avg_loss,
        threshold=threshold,
        per_instance=per_instance,
        y_true=all_true,
        y_prob=all_pred,
    )