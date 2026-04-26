from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import RDBInstance


def _joinable(row_a: dict[str, str], row_b: dict[str, str]) -> bool:
    for attr, value in row_a.items():
        if row_b.get(attr) == value:
            return True
    return False


def _attribute_counter(rows: list[dict[str, str]], idx: int) -> int:
    target = rows[idx]
    count = 0
    for attr, value in target.items():
        for j, other in enumerate(rows):
            if j == idx:
                continue
            if other.get(attr) == value:
                count += 1
                break
    return count


def _joinable_counter(rows: list[dict[str, str]], idx: int) -> int:
    target = rows[idx]
    total = 0
    for j, other in enumerate(rows):
        if j == idx:
            continue
        if _joinable(target, other):
            total += 1
    return total


def _labels_from_predictor(
    rows: list[dict[str, str]],
    predictor_name: str,
    threshold: int,
) -> list[int]:
    labels: list[int] = []
    for idx in range(len(rows)):
        if predictor_name == "joinable_counter_ge_n":
            value = _joinable_counter(rows, idx)
        elif predictor_name == "attribute_counter_ge_n":
            value = _attribute_counter(rows, idx)
        else:
            raise ValueError(f"Unknown predictor: {predictor_name}")
        labels.append(int(value >= threshold))
    return labels


def _copy_with_namespace(
    template_rows: list[dict[str, str]],
    namespace: str,
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for row in template_rows:
        copied.append({attr: f"{namespace}_{value}" for attr, value in row.items()})
    return copied


def _expand_template(
    template_rows: list[dict[str, str]],
    n_rows: int,
    task_prefix: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    block_size = len(template_rows)
    blocks = max(1, n_rows // block_size)

    for block_idx in range(blocks):
        rows.extend(_copy_with_namespace(template_rows, f"{task_prefix}b{block_idx}"))

    while len(rows) < n_rows:
        extra = _copy_with_namespace(template_rows, f"{task_prefix}e{len(rows)}")
        rows.extend(extra)

    return rows[:n_rows]


def _direct_template_variant_1() -> list[dict[str, str]]:
    # From the paper's Direct-favored style (Table 2 pattern).
    return [
        {"A1": "1", "A2": "2"},
        {"A1": "1", "A2": "1"},
        {"A1": "2", "A2": "1"},
        {"A1": "2", "A2": "2"},
        {"A1": "1", "A2": "1"},
        {"A1": "2", "A2": "2"},
    ]


def _direct_template_variant_2() -> list[dict[str, str]]:
    # Same structure with an extra attribute to create a second distinct Direct-favored task.
    return [
        {"A1": "1", "A2": "3", "A3": "9"},
        {"A1": "1", "A2": "1", "A3": "8"},
        {"A1": "2", "A2": "1", "A3": "8"},
        {"A1": "2", "A2": "3", "A3": "9"},
        {"A1": "1", "A2": "1", "A3": "7"},
        {"A1": "2", "A2": "3", "A3": "7"},
    ]


def _indirect_template_variant_1() -> list[dict[str, str]]:
    # From the paper's Indirect-favored style (Table 1 pattern).
    return [
        {"A1": "1", "A2": "2"},
        {"A1": "1", "A2": "2"},
        {"A1": "1", "A2": "3"},
    ]


def _indirect_template_variant_2() -> list[dict[str, str]]:
    # A second Indirect-favored pattern where Direct graph tends to collapse row colors.
    return [
        {"A1": "1", "A2": "2", "A3": "9"},
        {"A1": "1", "A2": "2", "A3": "8"},
        {"A1": "1", "A2": "3", "A3": "8"},
        {"A1": "1", "A2": "4", "A3": "7"},
    ]


def _build_task_rows(task_name: str, n_rows: int, instance_idx: int) -> list[dict[str, str]]:
    if task_name == "direct_better_task_1":
        return _expand_template(_direct_template_variant_1(), n_rows, f"d1_{instance_idx}")
    if task_name == "direct_better_task_2":
        return _expand_template(_direct_template_variant_2(), n_rows, f"d2_{instance_idx}")
    if task_name == "indirect_better_task_1":
        return _expand_template(_indirect_template_variant_1(), n_rows, f"i1_{instance_idx}")
    if task_name == "indirect_better_task_2":
        return _expand_template(_indirect_template_variant_2(), n_rows, f"i2_{instance_idx}")
    raise ValueError(f"Unknown task name: {task_name}")


def generate_four_generated_tasks(
    n_instances_per_task: int,
    n_rows: int,
    seed: int,
) -> dict[str, list[RDBInstance]]:
    """Generate 4 tasks: 2 expected to favor Direct and 2 expected to favor Indirect."""

    _ = seed  # deterministic template generation by design
    tasks: dict[str, list[RDBInstance]] = {
        "direct_better_task_1": [],
        "direct_better_task_2": [],
        "indirect_better_task_1": [],
        "indirect_better_task_2": [],
    }

    configs = [
        ("direct_better_task_1", "joinable_counter_ge_n", 4),
        ("direct_better_task_2", "joinable_counter_ge_n", 5),
        ("indirect_better_task_1", "attribute_counter_ge_n", 2),
        ("indirect_better_task_2", "attribute_counter_ge_n", 2),
    ]

    for task_name, predictor, threshold in configs:
        for idx in range(n_instances_per_task):
            rows = _build_task_rows(task_name, n_rows=n_rows, instance_idx=idx)
            labels = _labels_from_predictor(rows, predictor, threshold)

            tasks[task_name].append(
                RDBInstance(
                    instance_id=f"{task_name}_{idx:05d}",
                    task_name=task_name,
                    predictor_name=predictor,
                    threshold=threshold,
                    rows=rows,
                    labels=labels,
                    source="synthetic",
                )
            )

    return tasks


def load_real_rdb_instances(data_dir: Path) -> list[RDBInstance]:
    """Load real RDB instances from CSV row files and derive benchmark labels."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Real RDB directory not found: {data_dir}")

    instances: list[RDBInstance] = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    for file_path in csv_files:
        frame = pd.read_csv(file_path)
        if frame.empty:
            continue

        rows = [
            {col: str(value) for col, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]

        predictor_name = "joinable_counter_ge_n"
        threshold = max(1, min(5, len(rows) // 4))
        labels = _labels_from_predictor(rows, predictor_name, threshold)

        instances.append(
            RDBInstance(
                instance_id=file_path.stem,
                task_name="real_data_benchmark",
                predictor_name=predictor_name,
                threshold=threshold,
                rows=rows,
                labels=labels,
                source="real",
            )
        )

    return instances
