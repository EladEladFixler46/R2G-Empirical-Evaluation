from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import torch
import pandas as pd
import numpy as np

from r2g_eval.models import GraphInstance, RDBInstance


@dataclass(slots=True)
class BaseR2GAlgorithm(ABC):
    """
    Shared interface for all Relational-to-Graph (R2G) algorithms.
    """
    name: str
    
    def _run(self, instance: RDBInstance) -> GraphInstance:
        raise NotImplementedError


class DirectR2GAlgorithm(BaseR2GAlgorithm):
    """
    Direct Relational-to-Graph (R2G) conversion algorithm.
    Supports node-level embeddings from RDB column.
    """

    def __init__(self) -> None:
        super().__init__(name="r2g_direct")

    def _run(self, instance: RDBInstance) -> GraphInstance:
        df_list = []
        global_row_idx = 0
        node_to_id = {}

        for df_name, df in instance.data.items():
            temp_df = df.copy()
            n_rows = len(temp_df)
            temp_idx = np.arange(global_row_idx, global_row_idx + n_rows)
            temp_df['__global_idx'] = temp_idx
            node_to_id.update(dict(zip(temp_idx, temp_df['ID'].astype(str))))
            global_row_idx += n_rows
            df_list.append(temp_df)

        if not df_list:
            return GraphInstance(
                instance_id=instance.instance_id,
                task_name=instance.task_name,
                node_to_id={},
                embeddings=torch.zeros((0, 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                metadata={"num_nodes": 0}
            )

        combined_df = pd.concat(df_list, ignore_index=True)
        num_nodes = len(combined_df)

        if 'Embeddings' in combined_df.columns:
            emb_list = combined_df['Embeddings'].tolist()
            embeddings = torch.tensor(np.array(emb_list), dtype=torch.float)
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(-1)
        else:
            embeddings = torch.zeros((num_nodes, 1))

        cols_to_melt = [c for c in combined_df.columns if c not in ['ID', '__global_idx', 'Embeddings']]
        melted = combined_df.melt(id_vars=['__global_idx'], value_vars=cols_to_melt, 
                                  var_name='attribute', value_name='value')
        melted = melted.dropna(subset=['value'])

        merged = pd.merge(melted, melted, on=['attribute', 'value'])
        edges = merged[merged['__global_idx_x'] != merged['__global_idx_y']]
        edges = edges[['__global_idx_x', '__global_idx_y']].drop_duplicates()

        if len(edges) > 0:
            edge_index = torch.tensor(np.vstack((edges['__global_idx_x'].values, 
                                                 edges['__global_idx_y'].values)), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return GraphInstance(
            instance_id=instance.instance_id,
            task_name=instance.task_name,
            node_to_id=node_to_id,
            embeddings=embeddings,
            edge_index=edge_index,
            metadata={"num_nodes": num_nodes}
        )


class IndirectR2GAlgorithm(BaseR2GAlgorithm):
    """
    Indirect Relational-to-Graph (R2G) conversion algorithm.
    Supports node-level embeddings and value-node initialization flags.
    """

    def __init__(self, use_val_as_embedding: bool = False) -> None:
        super().__init__(name="r2g_indirect")
        self.use_val_as_embedding = use_val_as_embedding

    def _run(self, instance: RDBInstance) -> GraphInstance:
        df_list = []
        global_row_idx = 0
        node_to_id = {}

        for df_name, df in instance.data.items():
            temp_df = df.copy()
            n_rows = len(temp_df)
            temp_idx = np.arange(global_row_idx, global_row_idx + n_rows)
            temp_df['__global_idx'] = temp_idx
            node_to_id.update(dict(zip(temp_idx, temp_df['ID'].astype(str))))
            global_row_idx += n_rows
            df_list.append(temp_df)

        if not df_list:
            return GraphInstance(
                instance_id=instance.instance_id,
                task_name=instance.task_name,
                node_to_id={},
                embeddings=torch.zeros((0, 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                metadata={}
            )

        combined_df = pd.concat(df_list, ignore_index=True)
        num_row_nodes = len(combined_df)

        if 'Embeddings' in combined_df.columns:
            row_emb_list = combined_df['Embeddings'].tolist()
            row_embeddings = torch.tensor(np.array(row_emb_list), dtype=torch.float)
            if row_embeddings.dim() == 1:
                row_embeddings = row_embeddings.unsqueeze(-1)
        else:
            row_embeddings = torch.zeros((num_row_nodes, 1))
        
        emb_dim = row_embeddings.shape[1]

        cols_to_melt = [c for c in combined_df.columns if c not in ['ID', '__global_idx', 'Embeddings']]
        melted = combined_df.melt(id_vars=['__global_idx'], value_vars=cols_to_melt, 
                                  var_name='attribute', value_name='value')
        melted = melted.dropna(subset=['value'])

        melted['cv_str'] = melted['attribute'].astype(str) + "=" + melted['value'].astype(str)
        unique_cvs = melted['cv_str'].unique()
        num_cv_nodes = len(unique_cvs)
        
        cv_idx_range = np.arange(num_row_nodes, num_row_nodes + num_cv_nodes)
        cv_to_idx = dict(zip(unique_cvs, cv_idx_range))
        node_to_id.update(dict(zip(cv_idx_range, unique_cvs)))

        if self.use_val_as_embedding:
            try:
                val_data = [float(cv.split('=')[1]) for cv in unique_cvs]
            except (ValueError, IndexError):
                val_data = [0.0] * num_cv_nodes
            val_embeddings = torch.tensor(val_data, dtype=torch.float).view(-1, 1)
            if emb_dim > 1:
                padding = torch.zeros((num_cv_nodes, emb_dim - 1))
                val_embeddings = torch.cat([val_embeddings, padding], dim=1)
        else:
            val_embeddings = torch.ones((num_cv_nodes, emb_dim))

        full_embeddings = torch.cat([row_embeddings, val_embeddings], dim=0)
        melted['cv_idx'] = melted['cv_str'].map(cv_to_idx)

        if len(melted) > 0:
            all_src = np.concatenate([melted['__global_idx'].values, melted['cv_idx'].values])
            all_dst = np.concatenate([melted['cv_idx'].values, melted['__global_idx'].values])
            edge_index = torch.tensor(np.vstack((all_src, all_dst)), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return GraphInstance(
            instance_id=instance.instance_id,
            task_name=instance.task_name,
            node_to_id=node_to_id,
            embeddings=full_embeddings,
            edge_index=edge_index,
            metadata={"num_row_nodes": num_row_nodes, "num_cv_nodes": num_cv_nodes}
        )


def get_default_algorithms() -> list[BaseR2GAlgorithm]:
    return [DirectR2GAlgorithm(), IndirectR2GAlgorithm()]