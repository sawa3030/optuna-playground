import optuna

def objective1(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2

study = optuna.create_study(direction="minimize", storage="sqlite:///tmp.db")
study.optimize(objective1, n_trials=10)

def objective2(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    # y = trial.suggest_int("y", -5, 5)
    # z = trial.suggest_float("z", -5, 5)
    return x**2

study.optimize(objective2, n_trials=10)

