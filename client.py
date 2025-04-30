import optuna
from optuna.samplers import TPESampler
import numpy as np



def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x**2

    
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


# study = optuna.create_study(sampler=TPESampler(0.5, True, False, 10))
study = optuna.create_study(sampler=TPESampler(True, 0.5, True, False, 10, 24, default_gamma, default_weights, 10))
study.optimize(objective, n_trials=10)