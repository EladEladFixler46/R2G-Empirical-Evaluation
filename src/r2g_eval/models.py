from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
import torch


@dataclass(slots=True)
class RDBInstance:
    """Single relational-database instance used in experiments."""

    instance_id: str
    task_name: str
    data: dict[str, pd.DataFrame] #each df must contain a 'ID' column, and the rest of the columns are attributes


@dataclass(slots=True)
class GraphInstance:
    """Output for one algorithm on one graph instance."""

    instance_id: str
    task_name: str
    node_to_id: dict[int, str]
    embeddings: torch.Tensor
    edge_index: torch.Tensor
    metadata: dict[str, Any]

@dataclass(slots=True)
class ProblemInstance:
    """Defines a problem instance for evaluation, including the RDB instance and the expected graph properties."""
    
    instance_id: str
    task_name: str
    rdb_instance: RDBInstance
    expected_properties: dict[str, float] # must be the ID of the row and the expected value of the property for that node