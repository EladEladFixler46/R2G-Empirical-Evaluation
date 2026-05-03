from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import torch


@dataclass(slots=True)
class RDBInstance:
    """Single relational-database instance used in experiments."""

    instance_id: str
    task_name: str
    data: dict[str, pd.DataFrame] 
    fkeys: dict[str, dict[str, str]] = field(default_factory=dict)


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
    """Defines a problem instance for evaluation."""
    
    instance_id: str
    task_name: str
    rdb_instance: RDBInstance
    expected_properties: dict[str, float]