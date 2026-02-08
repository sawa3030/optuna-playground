from __future__ import annotations

import csv
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import optunahub
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)

N_REPEATS = 100
N_TRIALS_LIST = [25, 50, 75, 100]
BATCH_SIZES = [5, 10, 50]
CONSTANT_LIAR_LIST: list[Optional[str]] = [None, "best", "worst", "mean"]

BBOB_DIMS = [2]
BBOB_FUNCTION_IDS = [1,6,10,15,20]

HPOBENCH_DATASET_IDS = [0, 1, 2]

# N_REPEATS = 1
# N_TRIALS_LIST = [25, 50, 75, 100]
# BATCH_SIZES = [5, 10]
# CONSTANT_LIAR_LIST: list[Optional[str]] = [None, "best", "worst", "mean"]

# BBOB_DIMS = [2]
# BBOB_FUNCTION_IDS = [1]
# HPOBENCH_DATASET_IDS = [0]

# Output
OUT_DIR = Path("benchmark_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DETAILS_JSONL = OUT_DIR / "details.jsonl"
SUMMARY_CSV = OUT_DIR / "summary.csv"


def suggest_from_distribution(trial: optuna.Trial, name: str, dist: BaseDistribution) -> Any:
    if isinstance(dist, FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, log=dist.log, step=dist.step)
    if isinstance(dist, IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, log=dist.log, step=dist.step)
    if isinstance(dist, CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    raise TypeError(f"Unsupported distribution type for {name}: {type(dist)}")


def suggest_params(trial: optuna.Trial, search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
    return {name: suggest_from_distribution(trial, name, dist) for name, dist in search_space.items()}


def run_batched_study_checkpoints(
    problem: Any,
    sampler: optuna.samplers.BaseSampler,
    n_trials_max: int,
    batch_size: int,
    checkpoints: list[int],
) -> dict[int, float]:
    if n_trials_max < batch_size:
        raise ValueError("n_trials_max must be >= batch_size")
    if any(c > n_trials_max for c in checkpoints):
        raise ValueError("All checkpoints must be <= n_trials_max")

    study = optuna.create_study(directions=problem.directions, sampler=sampler)

    best_val: Optional[float] = None
    best_at: dict[int, float] = {}

    completed = 0
    checkpoints_sorted = sorted(set(checkpoints))
    next_idx = 0

    while completed < n_trials_max:
        current_batch = min(batch_size, n_trials_max - completed)

        trial_numbers: list[int] = []
        params_batch: list[dict[str, Any]] = []

        for _ in range(current_batch):
            t = study.ask()
            trial_numbers.append(t.number)
            params_batch.append(suggest_params(t, problem.search_space))

        values: list[Union[float, Sequence[float]]] = []
        for params in params_batch:
            values.append(problem.evaluate(params))

        for tn, v in zip(trial_numbers, values):
            study.tell(tn, v)
            completed += 1

            while next_idx < len(checkpoints_sorted) and completed == checkpoints_sorted[next_idx]:
                best_at[checkpoints_sorted[next_idx]] = study.best_trial.value
                next_idx += 1

            if completed >= n_trials_max:
                break

    return best_at


def iter_bbob_problems() -> Iterable[Tuple[str, Any]]:
    bbob = optunahub.load_module("benchmarks/bbob")
    for d in BBOB_DIMS:
        for function_id in BBOB_FUNCTION_IDS:
            p = bbob.Problem(function_id=function_id, dimension=d)
            yield (f"bbob:function{function_id}:dim{d}", p)


def iter_hpobench_problems() -> Iterable[Tuple[str, Any]]:
    hpobench = optunahub.load_module("benchmarks/hpobench_nn")
    for dataset_id in HPOBENCH_DATASET_IDS:
        p = hpobench.Problem(dataset_id=dataset_id)
        yield (f"hpobench_nn:dataset{dataset_id}", p)


@dataclass
class DetailRecord:
    benchmark: str
    constant_liar: Optional[str]
    n_trials: int
    batch_size: int
    repeat: int
    seed: int
    best_value: float


@dataclass
class SummaryRecord:
    benchmark: str
    constant_liar: Optional[str]
    n_trials: int
    batch_size: int
    n_repeats: int
    best_value_mean: float
    best_value_std: float


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    DETAILS_JSONL.write_text("", encoding="utf-8")
    SUMMARY_CSV.write_text("", encoding="utf-8")

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "constant_liar",
                "n_trials",
                "batch_size",
                "n_repeats",
                "best_value_mean",
                "best_value_std",
            ],
        )
        writer.writeheader()

    problems = list(iter_bbob_problems()) + list(iter_hpobench_problems())

    checkpoints = N_TRIALS_LIST
    n_trials_max = max(N_TRIALS_LIST)
    i = 0

    for benchmark_name, problem in problems:
        i += 1
        for batch_size in BATCH_SIZES:
            if n_trials_max < batch_size:
                continue

            for constant_liar in CONSTANT_LIAR_LIST:
                best_vals_by_n: dict[int, list[float]] = {n: [] for n in checkpoints}

                for r in range(N_REPEATS):
                    seed = i*1000 + r

                    sampler = optuna.samplers.GPSampler(constant_liar=constant_liar, seed=seed)

                    best_at = run_batched_study_checkpoints(
                        problem=problem,
                        sampler=sampler,
                        n_trials_max=n_trials_max,
                        batch_size=batch_size,
                        checkpoints=checkpoints,
                    )

                    for n in checkpoints:
                        best = float(best_at[n])
                        best_vals_by_n[n].append(best)

                        append_jsonl(
                            DETAILS_JSONL,
                            asdict(
                                DetailRecord(
                                    benchmark=benchmark_name,
                                    constant_liar=constant_liar,
                                    n_trials=n,
                                    batch_size=batch_size,
                                    repeat=r,
                                    seed=seed,
                                    best_value=best,
                                )
                            ),
                        )

                for n in checkpoints:
                    vals = best_vals_by_n[n]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                    summary = SummaryRecord(
                        benchmark=benchmark_name,
                        constant_liar=constant_liar,
                        n_trials=n,
                        batch_size=batch_size,
                        n_repeats=len(vals),
                        best_value_mean=mean,
                        best_value_std=std,
                    )

                    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "benchmark",
                                "constant_liar",
                                "n_trials",
                                "batch_size",
                                "n_repeats",
                                "best_value_mean",
                                "best_value_std",
                            ],
                        )
                        writer.writerow(asdict(summary))

                    print(
                        f"[OK] {benchmark_name} | liar={constant_liar} | batch={batch_size} | "
                        f"n_trials={n} | mean={mean:.6g} | std={std:.3g}"
                    )

    print(f"\nWrote:\n  - {DETAILS_JSONL}\n  - {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
