from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RDBInstance:
    """Single relational-database instance used in experiments."""

    instance_id: str
    task_name: str
    predictor_name: str
    threshold: int
    rows: list[dict[str, str]]
    labels: list[int]
    source: str


@dataclass(slots=True)
class AlgorithmResult:
    """Output for one algorithm on one graph instance."""

    algorithm: str
    instance_id: str
    task_name: str
    predictor_name: str
    threshold: int
    source: str
    wl_color: int
    objective_score: float
    runtime_sec: float
    metadata: dict[str, Any]
