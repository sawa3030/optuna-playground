import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2, (x - 2)**2 + (y - 2)**2


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
    x = trial.params["x"]
    y = trial.params["y"]
    return (x - 2, y - 2)


mode = ["single", "multi", "constr"][0]
if mode == "single":
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=0))
    obj_func = objective
elif mode == "multi":
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=0), directions=["minimize"]*2)
    obj_func = multi_objective
elif mode == "constr":
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=0, constraints_func=constraints))
    obj_func = objective

study.optimize(obj_func, n_trials=20)
trials = study.trials
print((trials[-1].datetime_complete - trials[0].datetime_start).total_seconds())