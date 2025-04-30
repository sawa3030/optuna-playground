# import optuna


# def objective(trial):
#     x = trial.suggest_float("x", -1, 1)
#     y = trial.suggest_int("y", -1, 1)
#     return x**2 + y


# sampler = optuna.samplers.CmaEsSampler()
# study = optuna.create_study(sampler=sampler)
# study.optimize(objective, n_trials=20)

# import optuna
# from optuna.visualization.matplotlib._matplotlib_imports import plt
# fig, axs = plt.subplots()

# print(type(axs))
# print(type(fig))

# import optuna


# def objective(trial):
#     x = trial.suggest_float("x", -100, 100)
#     y = trial.suggest_categorical("y", [-1, 0, 1])
#     z = trial.suggest_int("z", -100, 100)
#     return x**2 + y + z


# sampler = optuna.samplers.TPESampler(seed=10)
# study = optuna.create_study(sampler=sampler)
# study.optimize(objective, n_trials=30)

# optuna.visualization.matplotlib.plot_contour(study, params=["x", "y", "z"])

# from optuna.visualization.matplotlib._matplotlib_imports import _imports
# if _imports.is_successful():
#     from optuna.visualization.matplotlib._matplotlib_imports import Axes

import optuna
from optuna.samplers import TPESampler
import numpy as np
from optuna import create_study
from optuna import Trial

def default_gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_default_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def default_weights(x: int) -> np.ndarray:
    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x**2


study = optuna.create_study(sampler=TPESampler(1.0, True, False, 10, 24, default_gamma, default_weights, None))
study.optimize(objective, n_trials=10)
