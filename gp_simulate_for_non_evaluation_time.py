from __future__ import annotations

import argparse
import itertools
import pickle
import time
from pathlib import Path
from typing import Any

import optuna
import optunahub
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
import numpy as np

# Problem = optunahub.load_module("benchmarks/hpobench_nn").Problem
# Problem = optunahub.load_module("benchmarks/bbob").Problem
# wfg = optunahub.load_module("benchmarks/wfg")


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
    tau: float,
    use_qmc: bool,
) -> optuna.Study:
    if n_workers <= 0:
        raise ValueError("n_workers must be >= 1")

    # problem = Problem(dataset_id=dataset_id, metric_names=["val_acc"], seed=0)
    # problem = Problem(function_id=dataset_id, dimension=2)
    # problem = wfg.Problem(function_id=4, n_objectives=4, dimension=8)
    # problem = wfg.Problem(function_id=4, n_objectives=2, dimension=3, k=1)

    def objective(x: float, y: float) -> float:
        # return float(np.cos(2*x) * np.cos(y) + np.sin(x))
        return float(np.cos(x) + y)

    def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
        x = trial.params["x"]
        y = trial.params["y"]
        # c = float(np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) - 0.5)
        c = float(np.sin(x)*np.sin(y) + 0.95)
        return (c,)
        
    def feasible(trial: optuna.trial.FrozenTrial) -> bool:
        return all(c <= 0 for c in constraints(trial))

    sampler = optuna.samplers.GPSampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
        constraints_func=constraints,
    )
    sampler._tau = tau
    sampler._use_qmc = use_qmc
    # sampler._q_acqf_n_qmc_samples = 128
    study = optuna.create_study(sampler=sampler)
    start_time = time.perf_counter()

    pending: list[tuple[optuna.Trial, dict[str, Any]] | None] = [None] * n_workers
    n_suggested = 0
    n_completed = 0

    for worker_id in itertools.cycle(range(n_workers)):
        if n_completed >= n_trials:
            break

        if pending[worker_id] is not None:
            previous_trial, previous_params = pending[worker_id]
            # value = problem.evaluate(previous_params)
            value = objective(**previous_params)
            study.tell(previous_trial, value)
            # print(f"Trial {previous_trial.number}: cumtime = {previous_trial.user_attrs['cumtime']}, value = {value}, constraint = {constraints(previous_trial)}")
            pending[worker_id] = None
            n_completed += 1
            if n_completed >= n_trials:
                break

        if n_suggested < n_trials:
            trial = study.ask()
            # params = suggest_params(trial, problem.search_space)
            x = trial.suggest_float("x", 0.0, 2 * np.pi)
            y = trial.suggest_float("y", 0.0, 2 * np.pi)
            params = {"x": x, "y": y}
            # trial.set_user_attr("cumtime", time.perf_counter() - start_time)
            trial.set_user_attr("cumtime", trial._trial_id + 1)  # Simulate cumulative time as trial number + 1
            # print(f"Trial {trial.number}: cumtime = {trial.user_attrs['cumtime']}")
            pending[worker_id] = (trial, params)
            n_suggested += 1

    trials = [
        t
        for t in study.trials[n_startup_trials+n_workers-1:]
        if t.state == optuna.trial.TrialState.COMPLETE and "cumtime" in t.user_attrs and feasible(t)
    ]
    trials = trials[: max(0, n_trials - n_startup_trials - n_workers + 1)]

    new_study = optuna.create_study()
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
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--use_qmc", type=bool, default=True)
    args = parser.parse_args()

    study_list: list[optuna.Study] = []
    for seed in range(args.n_seeds):
        study = simulate(
            n_workers=args.n_workers,
            n_trials=args.n_trials,
            n_startup_trials=args.n_startup_trials,
            seed=seed,
            dataset_id=args.dataset_id,
            tau=args.tau,
            use_qmc=args.use_qmc,
        )
        study_list.append(study)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.pickle"

    with out_path.open("wb") as f:
        pickle.dump(study_list, f)

    print(f"Saved {len(study_list)} studies to {out_path}")


if __name__ == "__main__":
    main()
