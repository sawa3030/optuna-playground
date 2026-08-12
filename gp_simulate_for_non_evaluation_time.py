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

wfg = optunahub.load_module("benchmarks/wfg")
dtlz_constrained = optunahub.load_module("benchmarks/dtlz_constrained")


def create_problem(
    problem_module: str,
    function_id: int,
    n_objectives: int,
    dimension: int,
    k: int | None,
    constraint_type: int | None,
):
    if problem_module == "wfg":
        return wfg.Problem(
            function_id=function_id,
            n_objectives=n_objectives,
            dimension=dimension,
            k=k,
        )
    if problem_module == "dtlz_constrained":
        if constraint_type is None:
            raise ValueError("--constraint-type is required for dtlz_constrained.")
        return dtlz_constrained.Problem(
            function_id=function_id,
            n_objectives=n_objectives,
            constraint_type=constraint_type,
            dimension=dimension,
        )
    raise ValueError(f"Unsupported problem_module: {problem_module}")


def set_problem_attrs(
    study: optuna.Study,
    *,
    problem_module: str,
    function_id: int,
    n_objectives: int,
    dimension: int,
    k: int | None,
    constraint_type: int | None,
) -> None:
    study.set_user_attr("problem_module", problem_module)
    study.set_user_attr("function_id", function_id)
    study.set_user_attr("n_objectives", n_objectives)
    study.set_user_attr("dimension", dimension)
    if k is not None:
        study.set_user_attr("k", k)
    if constraint_type is not None:
        study.set_user_attr("constraint_type", constraint_type)


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
    problem_module: str,
    function_id: int,
    n_objectives: int,
    dimension: int,
    k: int | None,
    constraint_type: int | None,
) -> optuna.Study:
    if n_workers <= 0:
        raise ValueError("n_workers must be >= 1")

    problem = create_problem(
        problem_module=problem_module,
        function_id=function_id,
        n_objectives=n_objectives,
        dimension=dimension,
        k=k,
        constraint_type=constraint_type,
    )

    sampler_kwargs: dict[str, Any] = dict(
        n_startup_trials=n_startup_trials,
        seed=seed,
        deterministic_objective=True,
    )
    if hasattr(problem, "constraints_func"):
        sampler_kwargs["constraints_func"] = problem.constraints_func

    sampler = optuna.samplers.GPSampler(**sampler_kwargs)
    # sampler._q_acqf_n_qmc_samples = 128
    study = optuna.create_study(directions=problem.directions, sampler=sampler)
    start_time = time.perf_counter()

    pending: list[tuple[optuna.Trial, dict[str, Any]] | None] = [None] * n_workers
    n_suggested = 0
    n_completed = 0

    for worker_id in itertools.cycle(range(n_workers)):
        print(f"Worker {worker_id} is working..., n_suggested={n_suggested}, n_completed={n_completed}")
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
    set_problem_attrs(
        new_study,
        problem_module=problem_module,
        function_id=function_id,
        n_objectives=n_objectives,
        dimension=dimension,
        k=k,
        constraint_type=constraint_type,
    )
    return new_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", default="./gp_simulator_results_without_evaltime_n5_bbo_dataset10")
    parser.add_argument("--n-workers", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-startup-trials", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--problem-module",
        choices=("wfg", "dtlz_constrained"),
        default="wfg",
    )
    parser.add_argument("--function-id", type=int, default=4)
    parser.add_argument("--n-objectives", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--constraint-type", type=int)
    args = parser.parse_args()

    study_list: list[optuna.Study] = []
    for seed in range(args.n_seeds):
        study = simulate(
            n_workers=args.n_workers,
            n_trials=args.n_trials,
            n_startup_trials=args.n_startup_trials,
            seed=seed,
            problem_module=args.problem_module,
            function_id=args.function_id,
            n_objectives=args.n_objectives,
            dimension=args.dimension,
            k=args.k,
            constraint_type=args.constraint_type,
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
