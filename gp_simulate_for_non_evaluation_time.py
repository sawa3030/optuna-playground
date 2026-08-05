from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import optuna
import optunahub
from study_snapshot import dump_study_list
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

# Problem = optunahub.load_module("benchmarks/hpobench_nn").Problem
Problem = optunahub.load_module("benchmarks/bbob").Problem


def suggest_from_distribution(
    trial: optuna.Trial, name: str, dist: BaseDistribution
) -> Any:
    if isinstance(dist, FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, log=dist.log, step=dist.step)
    if isinstance(dist, IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, log=dist.log, step=dist.step)
    if isinstance(dist, CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    raise TypeError(f"Unsupported distribution type for {name}: {type(dist)}")


def suggest_params(
    trial: optuna.Trial, search_space: dict[str, BaseDistribution]
) -> dict[str, Any]:
    return {
        name: suggest_from_distribution(trial, name, dist)
        for name, dist in search_space.items()
    }


def simulate(
    n_workers: int,
    n_trials: int,
    n_startup_trials: int,
    seed: int,
    dataset_id: int,
) -> optuna.Study:
    if n_workers <= 0:
        raise ValueError("n_workers must be >= 1")

    # problem = Problem(dataset_id=dataset_id, metric_names=["val_acc"], seed=0)
    problem = Problem(function_id=dataset_id, dimension=2)

    sampler = optuna.samplers.GPSampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
    )
    # sampler._q_acqf_n_qmc_samples = 128
    study = optuna.create_study(directions=problem.directions, sampler=sampler)
    start_time = time.perf_counter()

    pending: list[tuple[optuna.Trial, dict[str, Any]] | None] = [None] * n_workers
    n_suggested = 0
    n_completed = 0

    for worker_id in itertools.cycle(range(n_workers)):
        if n_completed >= n_trials:
            break

        if pending[worker_id] is not None:
            previous_trial, previous_params = pending[worker_id]
            value = problem.evaluate(previous_params)
            study.tell(previous_trial, value)
            pending[worker_id] = None
            n_completed += 1
            if n_completed >= n_trials:
                break

        if n_suggested < n_trials:
            trial = study.ask()
            params = suggest_params(trial, problem.search_space)
            trial.set_user_attr("cumtime", time.perf_counter() - start_time)
            # print(f"Trial {trial.number}: cumtime = {trial.user_attrs['cumtime']}")
            pending[worker_id] = (trial, params)
            n_suggested += 1

    trials = [
        t
        for t in study.trials[n_startup_trials+n_workers-1:]
        if t.state == optuna.trial.TrialState.COMPLETE and "cumtime" in t.user_attrs
    ]
    trials = trials[: max(0, n_trials - n_startup_trials - n_workers + 1)]

    new_study = optuna.create_study(directions=problem.directions)
    new_study.add_trials(trials)
    return new_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", default="./gp_simulator_results_without_evaltime_n5_bbo_dataset10")
    parser.add_argument("--n-workers", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-startup-trials", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--dataset-id", type=int, default=10)
    args = parser.parse_args()

    study_list: list[optuna.Study] = []
    for seed in range(args.n_seeds):
        study = simulate(
            n_workers=args.n_workers,
            n_trials=args.n_trials,
            n_startup_trials=args.n_startup_trials,
            seed=seed,
            dataset_id=args.dataset_id,
        )
        study_list.append(study)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.pickle"

    dump_study_list(out_path, study_list)

    print(f"Saved {len(study_list)} studies to {out_path}")


if __name__ == "__main__":
    main()
