from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

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

# N_REPEATS = 1
N_REPEATS = 10
N_TRIALS_LIST = [25, 50, 75, 100]
BATCH_SIZES = [5, 10, 50]

CONSTANT_LIAR_LIST: list[Optional[str]] = ["None"]

WFG_DIM = 9
WFG_N_OBJECTIVES_LIST = [2, 4, 6, 8]
WFG_FUNCTION_IDS = [1, 2, 3, 4, 5]

DTLZ_DIM = 9
DTLZ_N_OBJECTIVES_LIST = [2, 4, 6, 8]
DTLZ_FUNCTION_IDS = [1, 2, 3, 4, 5]

# BBOB_BIOBJ_DIMS = [2, 3, 5]
# BBOB_BIOBJ_FUNCTION_IDS = [1, 2, 3, 4, 5]

OUT_DIR = Path("benchmark_results_mo_hv")
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


# def _default_bbob_biobj_reference_point(problem: Any) -> np.ndarray:
#     n_obj = len(problem.directions)
#     if n_obj != 2:
#         raise ValueError(f"BBOB biobj is expected to have 2 objectives, got {n_obj}.")
#     # Large positive point (assuming minimization).
#     return np.array([1e3, 1e3], dtype=float)


def get_reference_point(benchmark_name: str, problem: Any) -> np.ndarray:
    """Return reference point by benchmark family."""
    # WFG: use problem.reference_point() if available.
    if benchmark_name.startswith("wfg:"):
        if not hasattr(problem, "reference_point"):
            raise AttributeError("WFG problem does not have `reference_point` method.")
        rp = problem.reference_point
        return np.asarray(rp)

    # DTLZ: fixed [1, 1, ..., 1]
    if benchmark_name.startswith("dtlz:"):
        n_obj = len(problem.directions)
        return np.ones(n_obj, dtype=float)

    # BBOB biobj: arbitrary conservative point
    if benchmark_name.startswith("bbob_biobj:"):
        return _default_bbob_biobj_reference_point(problem)

    raise ValueError(f"Unknown benchmark family for reference point: {benchmark_name}")


def run_batched_study_hv_checkpoints(
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

    study = optuna.create_study(directions=problem.directions, sampler=sampler)

    hv_at: dict[int, float] = {}
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

            # Record HV at checkpoints.
            while next_idx < len(checkpoints_sorted) and completed == checkpoints_sorted[next_idx]:
                hv_at[checkpoints_sorted[next_idx]] = _compute_current_hv(study, reference_point)
                next_idx += 1

            if completed >= n_trials_max:
                break

    return hv_at


# -----------------------------
# Problem iterators
# -----------------------------


def iter_wfg_problems() -> Iterable[Tuple[str, Any]]:
    # https://hub.optuna.org/benchmarks/wfg/
    wfg = optunahub.load_module("benchmarks/wfg")
    for n_obj in WFG_N_OBJECTIVES_LIST:
        for function_id in WFG_FUNCTION_IDS:
            found = False
            for k in range(1, WFG_DIM+1):
                try:
                    p = wfg.Problem(function_id=function_id, dimension=WFG_DIM, n_objectives=n_obj, k=k)
                except AssertionError:
                    continue
                found = True
                yield (f"wfg:function{function_id}:dim{WFG_DIM}:obj{n_obj}", p)
                break
            if not found:
                raise ValueError(f"Could not create WFG problem with function_id={function_id}, dim={DTLZ_DIM}, obj={n_obj} for any k in [1, {WFG_DIM}].")


def iter_dtlz_problems() -> Iterable[Tuple[str, Any]]:
    # https://hub.optuna.org/benchmarks/dtlz/
    dtlz = optunahub.load_module("benchmarks/dtlz")
    for n_obj in DTLZ_N_OBJECTIVES_LIST:
        for function_id in DTLZ_FUNCTION_IDS:
            p = dtlz.Problem(function_id=function_id, dimension=DTLZ_DIM, n_objectives=n_obj)
            yield (f"dtlz:function{function_id}:dim{DTLZ_DIM}:obj{n_obj}", p)

# def iter_bbob_biobj_problems() -> Iterable[Tuple[str, Any]]:
#     # Module path is assumed to be benchmarks/bbob_biobj in OptunaHub.
#     # Adjust if your local hub module name differs.
#     bbob_biobj = optunahub.load_module("benchmarks/bbob_biobj")
#     for d in BBOB_BIOBJ_DIMS:
#         for function_id in BBOB_BIOBJ_FUNCTION_IDS:
#             p = bbob_biobj.Problem(function_id=function_id, dimension=d)
#             yield (f"bbob_biobj:function{function_id}:dim{d}", p)



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
    DETAILS_JSONL.write_text("", encoding="utf-8")
    SUMMARY_CSV.write_text("", encoding="utf-8")

    summary_fieldnames = [
        "benchmark",
        "constant_liar",
        "n_trials",
        "batch_size",
        "n_repeats",
        "hypervolume_mean",
        "hypervolume_std",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()

    # problems = list(iter_wfg_problems()) + list(iter_dtlz_problems()) + list(iter_bbob_biobj_problems())
    problems = list(iter_wfg_problems()) + list(iter_dtlz_problems())

    checkpoints = N_TRIALS_LIST
    n_trials_max = max(N_TRIALS_LIST)

    benchmark_index = 0
    for benchmark_name, problem in problems:
        benchmark_index += 1

        reference_point = get_reference_point(benchmark_name, problem)

        # Safety check: reference point dimension should match #objectives
        n_obj = len(problem.directions)
        if reference_point.shape != (n_obj,):
            raise ValueError(
                f"Reference point shape mismatch for {benchmark_name}: "
                f"ref_point.shape={reference_point.shape}, n_objectives={n_obj}"
            )

        print(f"[INFO] benchmark={benchmark_name} | ref_point={reference_point.tolist()}")

        for batch_size in BATCH_SIZES:
            if n_trials_max < batch_size:
                continue

            for constant_liar in CONSTANT_LIAR_LIST:
                hv_by_n: dict[int, list[float]] = {n: [] for n in checkpoints}

                for r in range(N_REPEATS):
                    seed = benchmark_index * 1000 + r

                    sampler = optuna.samplers.GPSampler(
                        seed=seed,
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