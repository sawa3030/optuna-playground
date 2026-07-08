from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import optuna
import optunahub

import argparse
from pathlib import Path
import pickle
import random
import time
import numpy as np
import optunahub

plot_target_over_time = optunahub.load_module(
    "visualization/plot_target_over_time"
).plot_target_over_time
# Problem = optunahub.load_module("benchmarks/hpobench_nn").Problem
# Problem = optunahub.load_module("benchmarks/bbob").Problem
AsyncOptBenchmarkSimulator = optunahub.load_local_module(
    package = "benchmarks/async_opt_simulator",
    registry_root = "/home/eri/pfn/optunahub-registry/package",
).AsyncOptBenchmarkSimulator

class Problem(optunahub.benchmarks.BaseProblem):
    def __init__(self):
        self.directions = ["minimize"]
        self.search_space = {
            "x": optuna.distributions.FloatDistribution(0.0, 2 * np.pi),
            "y": optuna.distributions.FloatDistribution(0.0, 2 * np.pi),
        }

    def directions(self) -> list[str]:
        return self.directions
    
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return self.search_space

    def evaluate(self, params: dict[str, float]) -> float:
        x = params["x"]
        y = params["y"]
        return float(np.cos(2*x) * np.cos(y) + np.sin(x))



def simulate(n_workers: int, seed: int, dataset_id: int) -> optuna.Study:
    sim = AsyncOptBenchmarkSimulator(n_workers=n_workers)
    # problem = Problem(dataset_id=dataset_id, metric_names=["val_acc"], seed=0)
    # problem = Problem(function_id=10, dimension=2)
    # runtime_func = Problem(dataset_id=dataset_id, metric_names=["train_time"], seed=0)

    def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
        x = trial.params["x"]
        y = trial.params["y"]
        c = float(np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y) - 0.5)
        return (c,)

    problem = Problem()

    sampler = optuna.samplers.GPSampler(
        n_startup_trials=(n_init := 10),
        seed=seed,
    )
    n_trials = 100
    study = optuna.create_study(sampler=sampler)
    random.seed(seed)
    sim.optimize(
        study=study, problem=problem, runtime_func=lambda t: 100.0, n_trials=n_trials,
    )
    trials = [
        t
        for t in study.trials[n_init :]
        if t.state == optuna.trial.TrialState.COMPLETE and "cumtime" in t.user_attrs
    ]
    trials = trials[:n_trials - n_init]

    new_study = optuna.create_study()
    new_study.add_trials(trials)
    return new_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset-id", type=int, default=2)
    parser.add_argument("--out-dir", default="./gp_simulator_results_n5_dataset2")
    args = parser.parse_args()

    study_list = []
    for seed in range(10):
        study_list.append(simulate(n_workers=5, seed=seed, dataset_id=args.dataset_id))

    # out_dir = Path("./gp_simulator_results_without_evaltime_n5_bbo_dataset15")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.pickle"

    with out_path.open("wb") as f:
        pickle.dump(study_list, f)
    