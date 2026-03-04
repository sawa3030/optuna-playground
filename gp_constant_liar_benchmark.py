from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import optunahub
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info


N_REPEATS = 10
N_TRIALS_LIST = [25, 50, 75, 100]
BATCH_SIZES = [5, 10, 50]
CONSTANT_LIAR_LIST: list[Optional[str]] = [None]

OUT_DIR = Path("benchmark_results_mo_constrained_hv")
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


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _compute_current_hv(study: optuna.Study, reference_point: np.ndarray) -> float:
    hv_info = _get_hypervolume_history_info(study, reference_point)
    if len(hv_info.values) == 0:
        return 0.0
    return float(hv_info.values[-1])


def get_reference_point(problem: Any) -> np.ndarray:
    # C2-DTLZ2 is based on DTLZ2, so using [1, 1, ..., 1] is a common simple choice.
    n_obj = len(problem.directions)
    return np.ones(n_obj, dtype=float)


def run_batched_study_hv_checkpoints(
    *,
    problem: Any,
    sampler: optuna.samplers.BaseSampler,
    n_trials_max: int,
    batch_size: int,
    checkpoints: list[int],
    reference_point: np.ndarray,
) -> dict[int, float]:
    if n_trials_max < batch_size:
        raise ValueError("n_trials_max must be >= batch_size")
    if any(c > n_trials_max for c in checkpoints):
        raise ValueError("All checkpoints must be <= n_trials_max")

    study = optuna.create_study(
        directions=problem.directions,
        sampler=sampler,
    )

    hv_at: dict[int, float] = {}
    completed = 0
    checkpoints_sorted = sorted(set(checkpoints))
    next_idx = 0

    while completed < n_trials_max:
        current_batch = min(batch_size, n_trials_max - completed)

        trials: list[optuna.Trial] = []
        values: list[Any] = []

        # ask phase
        for _ in range(current_batch):
            t = study.ask()
            trials.append(t)

        # evaluate phase
        for t in trials:
            params = suggest_params(t, problem.search_space)
            v = problem.evaluate(params)
            values.append(v)

        # tell phase
        for t, v in zip(trials, values):
            study.tell(t, v)
            completed += 1

            while next_idx < len(checkpoints_sorted) and completed == checkpoints_sorted[next_idx]:
                hv_at[checkpoints_sorted[next_idx]] = _compute_current_hv(study, reference_point)
                next_idx += 1

            if completed >= n_trials_max:
                break

    return hv_at


@dataclass
class DetailRecord:
    benchmark: str
    constant_liar: Optional[str]
    n_trials: int
    batch_size: int
    repeat: int
    seed: int
    hypervolume: float


@dataclass
class SummaryRecord:
    benchmark: str
    constant_liar: Optional[str]
    n_trials: int
    batch_size: int
    n_repeats: int
    hypervolume_mean: float
    hypervolume_std: float


def main() -> None:
    summary_fieldnames = [
        "benchmark",
        "constant_liar",
        "n_trials",
        "batch_size",
        "n_repeats",
        "hypervolume_mean",
        "hypervolume_std",
    ]

    with SUMMARY_CSV.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()

    cdtlz = optunahub.load_module("benchmarks/dtlz_constrained")
    problem = cdtlz.Problem(
        function_id=2,
        n_objectives=2,
        constraint_type=2,
        dimension=3,
    )
    benchmark_name = "c2dtlz2:dim3:obj2"

    checkpoints = N_TRIALS_LIST
    n_trials_max = max(N_TRIALS_LIST)

    reference_point = get_reference_point(problem)
    n_obj = len(problem.directions)
    if reference_point.shape != (n_obj,):
        raise ValueError(
            f"Reference point shape mismatch: "
            f"ref_point.shape={reference_point.shape}, n_objectives={n_obj}"
        )

    print(f"[INFO] benchmark={benchmark_name} | ref_point={reference_point.tolist()}")

    for batch_size in BATCH_SIZES:
        if n_trials_max < batch_size:
            continue

        for constant_liar in CONSTANT_LIAR_LIST:
            hv_by_n: dict[int, list[float]] = {n: [] for n in checkpoints}

            for r in range(N_REPEATS):
                seed = r

                sampler = optuna.samplers.GPSampler(
                    seed=seed,
                    constraints_func=problem.constraints_func,
                    deterministic_objective=True,
                )

                hv_at = run_batched_study_hv_checkpoints(
                    problem=problem,
                    sampler=sampler,
                    n_trials_max=n_trials_max,
                    batch_size=batch_size,
                    checkpoints=checkpoints,
                    reference_point=reference_point,
                )

                for n in checkpoints:
                    hv = float(hv_at[n])
                    hv_by_n[n].append(hv)

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
                                hypervolume=hv,
                            )
                        ),
                    )

            for n in checkpoints:
                vals = hv_by_n[n]
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                summary = SummaryRecord(
                    benchmark=benchmark_name,
                    constant_liar=constant_liar,
                    n_trials=n,
                    batch_size=batch_size,
                    n_repeats=len(vals),
                    hypervolume_mean=mean,
                    hypervolume_std=std,
                )

                with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                    writer.writerow(asdict(summary))

                print(
                    f"[OK] {benchmark_name} | liar={constant_liar} | batch={batch_size} | "
                    f"n_trials={n} | HV mean={mean:.6g} | std={std:.3g}"
                )

    print(f"\nWrote:\n  - {DETAILS_JSONL}\n  - {SUMMARY_CSV}")


if __name__ == "__main__":
    main()