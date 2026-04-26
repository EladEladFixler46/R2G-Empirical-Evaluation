from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from .algorithms import BaseR2GAlgorithm
from .models import AlgorithmResult, RDBInstance


def run_algorithms_on_instances(
    instances: list[RDBInstance],
    algorithms: list[BaseR2GAlgorithm],
) -> pd.DataFrame:
    """Run all algorithms on all instances and return a tidy result table."""

    rows: list[dict] = []

    for instance in instances:
        for algorithm in algorithms:
            result: AlgorithmResult = algorithm.evaluate(instance)
            rows.append(
                {
                    "algorithm": result.algorithm,
                    "instance_id": result.instance_id,
                    "task_name": result.task_name,
                    "predictor_name": result.predictor_name,
                    "threshold": result.threshold,
                    "source": result.source,
                    "wl_color": result.wl_color,
                    "objective_score": result.objective_score,
                    "runtime_sec": result.runtime_sec,
                    **{f"meta_{k}": v for k, v in result.metadata.items()},
                }
            )

    return pd.DataFrame(rows)


def summarize_task1(results: pd.DataFrame) -> pd.DataFrame:
    """Summary for generated-data validation against theoretical expectations."""

    summary = (
        results.groupby(["task_name", "algorithm"], as_index=False)
        .agg(
            mean_wl_color=("wl_color", "mean"),
            std_wl_color=("wl_color", "std"),
            mean_objective=("objective_score", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .fillna(0.0)
    )
    return summary


def summarize_task_winners(results: pd.DataFrame) -> pd.DataFrame:
    """Compute winning algorithm by objective score for each task."""

    grouped = (
        results.groupby(["task_name", "algorithm"], as_index=False)
        .agg(
            mean_objective=("objective_score", "mean"),
            mean_wl_color=("wl_color", "mean"),
        )
        .sort_values(
            ["task_name", "mean_objective", "mean_wl_color"],
            ascending=[True, False, False],
        )
    )

    winners = grouped.drop_duplicates(subset=["task_name"]).copy()
    winners = winners.rename(columns={"algorithm": "winner_algorithm"})
    return winners[["task_name", "winner_algorithm", "mean_objective", "mean_wl_color"]]


def summarize_task2_monte_carlo(results: pd.DataFrame) -> pd.DataFrame:
    """WL color distribution table for Monte Carlo synthetic runs."""

    distribution = (
        results.groupby(["algorithm", "wl_color"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = distribution.groupby("algorithm")["count"].transform("sum")
    distribution["probability"] = distribution["count"] / totals
    return distribution.sort_values(["algorithm", "wl_color"]).reset_index(drop=True)


def compare_on_real_data(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare algorithms on real RDB data and provide a significance test."""

    real_results = results[results["source"] == "real"].copy()
    if real_results.empty:
        raise ValueError("No real-data rows found in results")

    scoreboard = (
        real_results.groupby("algorithm", as_index=False)
        .agg(
            mean_objective=("objective_score", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            wl_color_diversity=("wl_color", "nunique"),
        )
        .sort_values("mean_objective", ascending=False)
        .reset_index(drop=True)
    )

    stats_rows: list[dict] = []
    algorithms = sorted(real_results["algorithm"].unique())
    if len(algorithms) == 2:
        a, b = algorithms
        a_scores = real_results.loc[
            real_results["algorithm"] == a, "objective_score"
        ]
        b_scores = real_results.loc[
            real_results["algorithm"] == b, "objective_score"
        ]

        t_stat, p_value = stats.ttest_ind(a_scores, b_scores, equal_var=False)
        stats_rows.append(
            {
                "comparison": f"{a} vs {b}",
                "test": "Welch_t_test",
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
        )

    return scoreboard, pd.DataFrame(stats_rows)


def write_outputs(output_dir: Path, filename: str, frame: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    frame.to_csv(target, index=False)
    return target
