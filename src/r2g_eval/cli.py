from __future__ import annotations

import argparse
from pathlib import Path

from .algorithms import get_default_algorithms
from .data import generate_four_generated_tasks, load_real_rdb_instances
from .experiments import (
    compare_on_real_data,
    run_algorithms_on_instances,
    summarize_task1,
    summarize_task2_monte_carlo,
    summarize_task_winners,
    write_outputs,
)


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-instances-per-task", type=int, default=200)
    parser.add_argument("--n-rows", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))


def run_task1(args: argparse.Namespace) -> None:
    task_map = generate_four_generated_tasks(
        n_instances_per_task=args.n_instances_per_task,
        n_rows=args.n_rows,
        seed=args.seed,
    )

    all_instances = [inst for instances in task_map.values() for inst in instances]
    results = run_algorithms_on_instances(all_instances, get_default_algorithms())
    summary = summarize_task1(results)
    winners = summarize_task_winners(results)

    write_outputs(args.output_dir, "task1_raw_results.csv", results)
    target = write_outputs(args.output_dir, "task1_summary.csv", summary)
    winners_target = write_outputs(args.output_dir, "task1_winners.csv", winners)
    print(f"Task 1 complete. Summary written to: {target}")
    print(f"Task 1 winners written to: {winners_target}")


def run_task2(args: argparse.Namespace) -> None:
    task_map = generate_four_generated_tasks(
        n_instances_per_task=args.n_instances_per_task,
        n_rows=args.n_rows,
        seed=args.seed,
    )

    all_instances = [inst for instances in task_map.values() for inst in instances]
    results = run_algorithms_on_instances(all_instances, get_default_algorithms())
    distribution = summarize_task2_monte_carlo(results)

    write_outputs(args.output_dir, "task2_raw_results.csv", results)
    target = write_outputs(
        args.output_dir, "task2_wl_color_distribution.csv", distribution
    )
    print(f"Task 2 complete. WL distribution written to: {target}")


def run_task3(args: argparse.Namespace) -> None:
    instances = load_real_rdb_instances(args.real_data_dir)
    results = run_algorithms_on_instances(instances, get_default_algorithms())
    scoreboard, significance = compare_on_real_data(results)

    write_outputs(args.output_dir, "task3_raw_results.csv", results)
    score_target = write_outputs(args.output_dir, "task3_scoreboard.csv", scoreboard)
    sig_target = write_outputs(args.output_dir, "task3_significance.csv", significance)
    print(f"Task 3 complete. Scoreboard: {score_target}")
    print(f"Task 3 significance tests: {sig_target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r2g-eval",
        description="Empirical evaluation of two R2G algorithms",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    task1 = subparsers.add_parser(
        "task1",
        help="Generated-data experiment on 4 tasks (2 Direct-better, 2 Indirect-better)",
    )
    add_common_generation_args(task1)
    task1.set_defaults(func=run_task1)

    task2 = subparsers.add_parser(
        "task2",
        help="Monte Carlo simulation for WL color distributions on generated tasks",
    )
    add_common_generation_args(task2)
    task2.set_defaults(func=run_task2)

    task3 = subparsers.add_parser(
        "task3",
        help="Run both algorithms on real RDB data and compare",
    )
    task3.add_argument(
        "--real-data-dir",
        type=Path,
        default=Path("data/real_rdb"),
        help="Folder with CSV edge lists (source,target columns)",
    )
    task3.add_argument("--output-dir", type=Path, default=Path("results"))
    task3.set_defaults(func=run_task3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
