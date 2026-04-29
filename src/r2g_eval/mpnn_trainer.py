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


# ──────────────────────────────────────────────
# 1.  MPNN Layer
# ──────────────────────────────────────────────

class MPNNLayer(MessagePassing):
    """
    Single message-passing layer.
    Aggregation: sum over neighbours.
    Update:      sigmoid squashing to prevent value explosion.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")          
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.activation = nn.PReLU()          

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(torch.cat([x_i, x_j], dim=-1)))

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:          
        return torch.sigmoid(aggr_out)


# ──────────────────────────────────────────────
# 2.  Full MPNN (Configurable Layers)
# ──────────────────────────────────────────────

class MPNN(nn.Module):
    """
    Configurable MPNN that ends with a per-node scalar prediction (sigmoid output).
    hidden_dim controls the width of all intermediate representations.
    """

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
        return torch.sigmoid(self.head(x)).squeeze(-1)   


# ──────────────────────────────────────────────
# 3.  Helper: build a PyG Data object
# ──────────────────────────────────────────────

def _graph_instance_to_pyg(gi: GraphInstance) -> Data:
    """Convert a GraphInstance to a torch_geometric.data.Data object."""
    return Data(
        x=gi.embeddings.float(),
        edge_index=gi.edge_index,
        num_nodes=gi.embeddings.shape[0],
    )


def _problem_to_labelled_pyg(
    pi: ProblemInstance,
    algorithm: BaseR2GAlgorithm,
) -> tuple[Data, dict[int, float]]:
    """
    Run the R2G algorithm on a ProblemInstance's RDB data and attach
    node-level labels as a tensor aligned with node indices.
    """
    gi = algorithm._run(pi.rdb_instance)
    data = _graph_instance_to_pyg(gi)

    y = torch.full((data.num_nodes,), float("nan"))
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    id_to_node: dict[str, int] = {v: k for k, v in gi.node_to_id.items()}

    if pi.expected_properties:
        # Vectorized lookup instead of dict item loop
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


# ──────────────────────────────────────────────
# 4.  Trained model wrapper
# ──────────────────────────────────────────────

@dataclass
class TrainedMPNNModel:
    """
    Wraps a trained MPNN together with the R2G algorithm used during training.
    Exposes a simple `.predict(rdb_instance)` method.
    """

    mpnn: MPNN
    algorithm: BaseR2GAlgorithm
    device: torch.device

    def predict(self, rdb_instance: RDBInstance) -> dict[str, float]:
        """
        Convert an RDBInstance to a graph, run the MPNN, and return
        a dict mapping row ID -> predicted probability.
        """
        gi = self.algorithm._run(rdb_instance)
        data = _graph_instance_to_pyg(gi).to(self.device)

        self.mpnn.eval()
        with torch.no_grad():
            preds = self.mpnn(data.x, data.edge_index)   

        return {
            gi.node_to_id[node_idx]: preds[node_idx].item()
            for node_idx in gi.node_to_id
        }


# ──────────────────────────────────────────────
# 5.  Training
# ──────────────────────────────────────────────

@dataclass
class TrainingConfig:
    hidden_dim:  int   = 32
    num_layers:  int   = 3
    lr:          float = 1e-2
    epochs:      int   = 50
    batch_size:  int   = 32     # Added parameter for DataLoader
    device:      str   = "cpu"


def train(
    train_problems: list[ProblemInstance],
    test_problems:  list[ProblemInstance],
    algorithm:      BaseR2GAlgorithm,
    config:         TrainingConfig | None = None,
) -> tuple[TrainedMPNNModel, dict[str, Any]]:
    """
    Train an MPNN with SGD on the supplied training problems using fast PyG DataLoaders.
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)

    print("Pre-converting training graphs ...")
    train_data = [_problem_to_labelled_pyg(p, algorithm)[0] for p in train_problems]
    print("Pre-converting test graphs ...")
    test_data  = [_problem_to_labelled_pyg(p, algorithm)[0] for p in test_problems]

    # Convert to DataLoaders for massive speedup via batching
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    mpnn = MPNN(
        in_channels=1, 
        hidden_dim=config.hidden_dim, 
        num_layers=config.num_layers
    ).to(device)
    
    optimiser = torch.optim.SGD(mpnn.parameters(), lr=config.lr)
    loss_fn   = nn.BCELoss()

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
            preds  = mpnn(data.x, data.edge_index)
            y_true = data.y[data.labelled_mask]
            y_pred = preds[data.labelled_mask]
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
                preds  = mpnn(data.x, data.edge_index)
                y_true = data.y[data.labelled_mask]
                y_pred = preds[data.labelled_mask]
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


# ──────────────────────────────────────────────
# 6.  Evaluation
# ──────────────────────────────────────────────

@dataclass
class EvaluationResults:
    """Rich evaluation report for a trained model on a set of problems."""

    accuracy:          float
    precision:         float
    recall:            float
    f1:                float
    auc_roc:           float
    avg_loss:          float
    threshold:         float
    per_instance:      list[dict[str, Any]] = field(default_factory=list)

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
) -> EvaluationResults:
    """
    Evaluate a TrainedMPNNModel on a list of ProblemInstances.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )

    device   = model.device
    loss_fn  = nn.BCELoss()

    all_true:  list[float] = []
    all_pred:  list[float] = []
    per_instance: list[dict[str, Any]] = []

    model.mpnn.eval()
    with torch.no_grad():
        for pi in problems:
            gi   = model.algorithm._run(pi.rdb_instance)
            data = _graph_instance_to_pyg(gi).to(device)

            preds: torch.Tensor = model.mpnn(data.x, data.edge_index)

            id_to_node: dict[str, int] = {v: k for k, v in gi.node_to_id.items()}

            if not pi.expected_properties:
                continue

            # Vectorized numpy extraction
            keys = np.array(list(pi.expected_properties.keys()))
            vals = np.array(list(pi.expected_properties.values()))
            
            valid_mask = np.isin(keys, list(id_to_node.keys()))
            valid_keys = keys[valid_mask]
            valid_vals = vals[valid_mask]

            if len(valid_keys) == 0:
                continue
                
            indices = [id_to_node[k] for k in valid_keys]
            inst_true = valid_vals.tolist()
            inst_pred = preds[indices].tolist()

            t_arr = torch.tensor(inst_true)
            p_arr = torch.tensor(inst_pred)
            inst_loss = loss_fn(p_arr, t_arr).item()

            bin_pred = [1 if p >= threshold else 0 for p in inst_pred]

            per_instance.append({
                "instance_id":  pi.instance_id,
                "loss":         inst_loss,
                "n_nodes":      len(inst_true),
                "n_positive":   int(sum(inst_true)),
                "n_predicted_positive": int(sum(bin_pred)),
            })

            all_true.extend(inst_true)
            all_pred.extend(inst_pred)

    if not all_true:
        return EvaluationResults(
            accuracy=0, precision=0, recall=0, f1=0,
            auc_roc=0, avg_loss=0, threshold=threshold,
            per_instance=per_instance,
        )

    y_true = np.array(all_true)
    y_prob = np.array(all_pred)
    y_bin  = (y_prob >= threshold).astype(int)

    avg_loss  = loss_fn(torch.tensor(y_prob), torch.tensor(y_true)).item()
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
    )