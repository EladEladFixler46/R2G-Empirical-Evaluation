from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import networkx as nx

from .models import AlgorithmResult, RDBInstance


@dataclass(slots=True)
class BaseR2GAlgorithm:
    """Shared interface for all R2G algorithms under comparison."""

    name: str

    def evaluate(self, instance: RDBInstance) -> AlgorithmResult:
        start = perf_counter()
        wl_color, objective, metadata = self._run(instance)
        elapsed = perf_counter() - start
        return AlgorithmResult(
            algorithm=self.name,
            instance_id=instance.instance_id,
            task_name=instance.task_name,
            predictor_name=instance.predictor_name,
            threshold=instance.threshold,
            source=instance.source,
            wl_color=wl_color,
            objective_score=float(objective),
            runtime_sec=elapsed,
            metadata=metadata,
        )

    def _run(self, instance: RDBInstance) -> tuple[int, float, dict]:
        raise NotImplementedError


def _joinable(row_a: dict[str, str], row_b: dict[str, str]) -> bool:
    for attr, value in row_a.items():
        if row_b.get(attr) == value:
            return True
    return False


def _best_color_accuracy(row_colors: dict[str, int], labels: list[int]) -> float:
    groups: dict[int, list[int]] = {}
    for row_idx, color in row_colors.items():
        idx = int(row_idx.split(":", maxsplit=1)[1])
        groups.setdefault(color, []).append(labels[idx])

    correct = 0
    total = len(labels)
    for bucket in groups.values():
        positives = sum(bucket)
        negatives = len(bucket) - positives
        correct += max(positives, negatives)

    return correct / total if total else 0.0


def _wl_refinement(
    graph: nx.Graph,
    initial_labels: dict[str, str],
    rounds: int = 3,
) -> dict[str, int]:
    current = initial_labels.copy()

    for _ in range(rounds):
        signatures: dict[str, tuple[str, tuple[str, ...]]] = {}
        for node in graph.nodes:
            neighbor_colors = sorted(current[nbr] for nbr in graph.neighbors(node))
            signatures[node] = (current[node], tuple(neighbor_colors))

        ordered_unique = sorted(set(signatures.values()))
        remap = {sig: str(i) for i, sig in enumerate(ordered_unique)}
        current = {node: remap[sig] for node, sig in signatures.items()}

    color_vocab = {color: i for i, color in enumerate(sorted(set(current.values())))}
    return {node: color_vocab[color] for node, color in current.items()}


class DirectR2GAlgorithm(BaseR2GAlgorithm):
    """A Direct R2G: row-node graph with edges between joinable rows."""

    def __init__(self) -> None:
        super().__init__(name="r2g_direct")

    def _run(self, instance: RDBInstance) -> tuple[int, float, dict]:
        rows = instance.rows
        if not rows:
            return 0, 0.0, {"reason": "empty_graph"}

        graph = nx.Graph()
        row_nodes = [f"r:{i}" for i in range(len(rows))]
        graph.add_nodes_from(row_nodes)

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if _joinable(rows[i], rows[j]):
                    graph.add_edge(f"r:{i}", f"r:{j}")

        initial = {node: "row" for node in row_nodes}
        wl = _wl_refinement(graph, initial_labels=initial, rounds=3)
        row_colors = {node: wl[node] for node in row_nodes}

        unique_labels = len(set(row_colors.values()))
        objective = _best_color_accuracy(row_colors, instance.labels)
        wl_color = max(1, unique_labels)
        metadata = {
            "representation": "direct",
            "row_count": len(rows),
            "unique_row_wl_colors": unique_labels,
        }
        return wl_color, objective, metadata


class IndirectR2GAlgorithm(BaseR2GAlgorithm):
    """A Indirect R2G: row nodes + attribute-value nodes (bipartite)."""

    def __init__(self) -> None:
        super().__init__(name="r2g_indirect")

    def _run(self, instance: RDBInstance) -> tuple[int, float, dict]:
        rows = instance.rows
        if not rows:
            return 0, 0.0, {"reason": "empty_graph"}

        graph = nx.Graph()
        row_nodes = [f"r:{i}" for i in range(len(rows))]
        graph.add_nodes_from(row_nodes)

        av_nodes: set[str] = set()
        for i, row in enumerate(rows):
            row_node = f"r:{i}"
            for attr, value in row.items():
                av = f"av:{attr}:{value}"
                av_nodes.add(av)
                graph.add_edge(row_node, av)

        initial: dict[str, str] = {node: "row" for node in row_nodes}
        for node in av_nodes:
            initial[node] = "attr_value"

        wl = _wl_refinement(graph, initial_labels=initial, rounds=3)
        row_colors = {node: wl[node] for node in row_nodes}

        unique_labels = len(set(row_colors.values()))
        objective = _best_color_accuracy(row_colors, instance.labels)
        wl_color = max(1, unique_labels)
        metadata = {
            "representation": "indirect",
            "row_count": len(rows),
            "attr_value_nodes": len(av_nodes),
            "unique_row_wl_colors": unique_labels,
        }
        return wl_color, objective, metadata


def get_default_algorithms() -> list[BaseR2GAlgorithm]:
    """Return the two R2G implementations used in the experiments."""

    return [DirectR2GAlgorithm(), IndirectR2GAlgorithm()]
