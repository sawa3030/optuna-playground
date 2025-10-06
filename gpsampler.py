import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    z = trial.suggest_float("z", -5, 5)
    return x**2 + y**2 + z

study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=0))
obj_func = objective

study.optimize(obj_func, n_trials=20)
trials = study.trials
# print((trials[-1].datetime_complete - trials[0].datetime_start).total_seconds())