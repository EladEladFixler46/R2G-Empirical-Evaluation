from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from time import perf_counter

from models import GraphInstance, RDBInstance


@dataclass(slots=True)
class BaseR2GAlgorithm(ABC):
    """
    Shared interface for all Relational-to-Graph (R2G) algorithms.
    
    Attributes:
        name (str): The unique identifier name of the algorithm.
    """
    name: str
    
    def _run(self, instance: RDBInstance) -> GraphInstance:
        """
        Convert a relational database instance into a graph instance.
        
        Args:
            instance (RDBInstance): The input relational database data.
            
        Returns:
            GraphInstance: The resulting graph.
        """
        raise NotImplementedError


class DirectR2GAlgorithm(BaseR2GAlgorithm):
    """
    Direct Relational-to-Graph (R2G) conversion algorithm.
    """

    def __init__(self) -> None:
        super().__init__(name="r2g_direct")

    def _run(self, instance: RDBInstance) -> GraphInstance:
        import torch
        import pandas as pd
        import numpy as np

        df_list = []
        global_row_idx = 0
        node_to_id = {}

        for df_name, df in instance.data.items():
            temp_df = df.copy()
            n_rows = len(temp_df)
            
            # Numpy arange generates sequential indices instantly
            temp_idx = np.arange(global_row_idx, global_row_idx + n_rows)
            temp_df['__global_idx'] = temp_idx
            
            # Fast dictionary mapping for IDs
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
        cols_to_melt = [c for c in combined_df.columns if c not in ['ID', '__global_idx']]
        
        # Pandas melt vectorization: converts rows to (idx, attribute, value) instantly
        melted = combined_df.melt(id_vars=['__global_idx'], value_vars=cols_to_melt, 
                                  var_name='attribute', value_name='value')
        melted = melted.dropna(subset=['value'])

        # Pandas merge: vectorized relational join perfectly replaces the O(N^2) loops 
        # for finding rows that share the same value in the same column
        merged = pd.merge(melted, melted, on=['attribute', 'value'])
        
        # Boolean indexing to remove self-loops efficiently
        edges = merged[merged['__global_idx_x'] != merged['__global_idx_y']]
        
        # Drop duplicates in case multiple columns matched between the same two rows
        edges = edges[['__global_idx_x', '__global_idx_y']].drop_duplicates()

        if len(edges) > 0:
            edges_src = edges['__global_idx_x'].values
            edges_dst = edges['__global_idx_y'].values
            # Numpy vstack pushes arrays directly to torch, avoiding loops
            edge_index = torch.tensor(np.vstack((edges_src, edges_dst)), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        embeddings = torch.zeros((num_nodes, 1))

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
    Optimized with pandas and numpy vectorization.
    """

    def __init__(self) -> None:
        super().__init__(name="r2g_indirect")

    def _run(self, instance: RDBInstance) -> GraphInstance:
        import torch
        import pandas as pd
        import numpy as np

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
        cols_to_melt = [c for c in combined_df.columns if c not in ['ID', '__global_idx']]
        
        # Pandas melt vectorization
        melted = combined_df.melt(id_vars=['__global_idx'], value_vars=cols_to_melt, 
                                  var_name='attribute', value_name='value')
        melted = melted.dropna(subset=['value'])

        # Pandas string vectorization: concatenate columns rapidly without loops
        melted['cv_str'] = melted['attribute'].astype(str) + "=" + melted['value'].astype(str)
        
        # Numpy unique extracts all distinct Attribute-Value pairs instantly
        unique_cvs = melted['cv_str'].unique()
        num_cv_nodes = len(unique_cvs)
        
        cv_idx_range = np.arange(num_row_nodes, num_row_nodes + num_cv_nodes)
        cv_to_idx = dict(zip(unique_cvs, cv_idx_range))
        node_to_id.update(dict(zip(cv_idx_range, unique_cvs)))

        # Pandas map assigns the new node IDs across all rows efficiently
        melted['cv_idx'] = melted['cv_str'].map(cv_to_idx)

        if len(melted) > 0:
            edges_src = melted['__global_idx'].values
            edges_dst = melted['cv_idx'].values
            
            # Numpy concatenate to create undirected edges (src->dst and dst->src)
            all_src = np.concatenate([edges_src, edges_dst])
            all_dst = np.concatenate([edges_dst, edges_src])
            edge_index = torch.tensor(np.vstack((all_src, all_dst)), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        total_nodes = num_row_nodes + num_cv_nodes
        embeddings = torch.zeros((total_nodes, 1))

        return GraphInstance(
            instance_id=instance.instance_id,
            task_name=instance.task_name,
            node_to_id=node_to_id,
            embeddings=embeddings,
            edge_index=edge_index,
            metadata={"num_row_nodes": num_row_nodes, "num_cv_nodes": num_cv_nodes}
        )


def get_default_algorithms() -> list[BaseR2GAlgorithm]:
    """
    Retrieve the standard set of R2G implementations for evaluation.
    
    Returns:
        list[BaseR2GAlgorithm]: A list containing the Direct and Indirect algorithm instances.
    """
    return [DirectR2GAlgorithm(), IndirectR2GAlgorithm()]