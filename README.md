# R2G-Empirical-Evaluation

Practical framework for empirical evaluation of two R2G algorithms as part of a master thesis.

This repository is organized around three goals:

1. Validate that practical behavior on generated data follows the theoretical expectations.
2. Run Monte Carlo simulations over many RDB instances and analyze WL color outcomes for both R2G algorithms.
3. Run both algorithms on real RDB data and compare which one performs better.

## What is implemented

- Two algorithm implementations under a shared interface:
	- `r2g_direct` (Definition 3.1 Direct R2G)
	- `r2g_indirect` (Definition 3.1 Indirect R2G)
- Synthetic RDB generation based on paper-style counterexample templates.
- Four generated tasks:
	- 2 tasks where Direct should be better
	- 2 tasks where Indirect should be better
- The two row predictors from the theoretical section:
	- `rpred Joinable Counter>=N`
	- `rpred Attribute Counter>=N`
- Experiment pipelines for generated data and real RDB data.
- CSV outputs for reproducible reporting and plotting.

## Project Structure

```
.
|-- data/
|   `-- real_rdb/
|       `-- example_network.csv
|-- src/
|   `-- r2g_eval/
|       |-- algorithms.py
|       |-- cli.py
|       |-- data.py
|       |-- experiments.py
|       `-- models.py
|-- pyproject.toml
`-- README.md
```

## Setup

Use Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Task 1: Generated Data vs Theory (4 tasks)

Run controlled experiments on four generated tasks aligned with the paper.

```powershell
r2g-eval task1 --n-instances-per-task 300 --n-rows 60 --seed 123
```

Outputs:
- `results/task1_raw_results.csv`
- `results/task1_summary.csv`
- `results/task1_winners.csv`

`task1_winners.csv` should show:

- 2 tasks won by `r2g_direct`
- 2 tasks won by `r2g_indirect`

Winner selection is based on mean objective score, with mean WL row-color count as tie-breaker.

## Task 2: Monte Carlo WL Color Distribution

Run large-scale simulation and estimate WL color probabilities.

```powershell
r2g-eval task2 --n-instances-per-task 5000 --n-rows 60 --seed 321
```

Outputs:
- `results/task2_raw_results.csv`
- `results/task2_wl_color_distribution.csv`

The distribution file includes:
- `algorithm`
- `wl_color`
- `count`
- `probability`

## Task 3: Real RDB Benchmark

Place one or more CSV files in `data/real_rdb/` where each row is a database tuple and each column is an attribute.

Then run:

```powershell
r2g-eval task3 --real-data-dir data/real_rdb
```

Outputs:
- `results/task3_raw_results.csv`
- `results/task3_scoreboard.csv`
- `results/task3_significance.csv`

The significance file uses a Welch t-test between the two algorithms over real-data objective scores.

## Thesis Adaptation Notes

For final research results, adapt the implementation in `src/r2g_eval/algorithms.py` to match your exact theoretical R2G definitions and objective functions.

Recommended process:

1. Keep the pipeline fixed.
2. Replace algorithm internals only.
3. Re-run task 1 and verify theoretical alignment.
4. Scale task 2 for stable Monte Carlo estimates.
5. Run task 3 and report practical winner using scoreboard + significance.