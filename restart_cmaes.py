import optuna
import optunahub

import optunahub

module = optunahub.load_local_module(
    package="samplers/restart_cmaes",
    registry_root="../optunahub-registry/package",
)

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)

