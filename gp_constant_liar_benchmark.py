from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import optuna


N_REPEATS = 100
N_TRIALS_LIST = [25, 50, 75, 100]
BATCH_SIZES = [5, 10, 50]
CONSTANT_LIAR_LIST: list[Optional[str]] = ["none"]

# Output
OUT_DIR = Path("benchmark_results_constrained")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DETAILS_JSONL = OUT_DIR / "details.jsonl"
SUMMARY_CSV = OUT_DIR / "summary.csv"


def objective(x: float, y: float) -> float:
    return float(np.sin(x) + y)


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    x = trial.params["x"]
    y = trial.params["y"]
    c = float(np.sin(x) * np.sin(y) + 0.95)
    return (c,)


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


def run_batched_study_checkpoints(
    *,
    sampler: optuna.samplers.BaseSampler,
    n_trials_max: int,
    batch_size: int,
    checkpoints: list[int],
) -> dict[int, float]:
    if n_trials_max < batch_size:
        raise ValueError("n_trials_max must be >= batch_size")
    if any(c > n_trials_max for c in checkpoints):
        raise ValueError("All checkpoints must be <= n_trials_max")

    study = optuna.create_study(sampler=sampler)

    best_at: dict[int, float] = {}
    completed = 0

    checkpoints_sorted = sorted(set(checkpoints))
    next_idx = 0

    while completed < n_trials_max:
        current_batch = min(batch_size, n_trials_max - completed)

        trials: list[optuna.Trial] = []
        values: list[float] = []

        # ask phase
        for _ in range(current_batch):
            trial = study.ask()
            trials.append(trial)

        # evaluate phase
        for trial in trials:
            x = trial.suggest_float("x", 0.0, 2 * np.pi)
            y = trial.suggest_float("y", 0.0, 2 * np.pi)
            values.append(objective(x, y))

        # tell phase
        for trial, value in zip(trials, values):
            study.tell(trial, value)
            completed += 1

            while next_idx < len(checkpoints_sorted) and completed == checkpoints_sorted[next_idx]:
                try:
                    best_at[checkpoints_sorted[next_idx]] = float(study.best_trial.value)
                except ValueError:
                    best_at[checkpoints_sorted[next_idx]] = float("nan")
                next_idx += 1

            if completed >= n_trials_max:
                break

    return best_at


def main() -> None:
    # DETAILS_JSONL.write_text("", encoding="utf-8")

    with SUMMARY_CSV.open(mode="a", newline="", encoding="utf-8") as f:
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

    checkpoints = N_TRIALS_LIST
    n_trials_max = max(N_TRIALS_LIST)
    benchmark_name = "bench2"

    for batch_size in BATCH_SIZES:
        if n_trials_max < batch_size:
            continue

        for constant_liar in CONSTANT_LIAR_LIST:
            best_vals_by_n: dict[int, list[float]] = {n: [] for n in checkpoints}

            for r in range(N_REPEATS):
                seed = r

                sampler = optuna.samplers.GPSampler(
                    seed=seed,
                    constraints_func=constraints,
                    # 必要なら version に応じて constant_liar 関連設定をここに追加
                    # 例: constant_liar=True
                )

                best_at = run_batched_study_checkpoints(
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