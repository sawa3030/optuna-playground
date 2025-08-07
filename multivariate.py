import optuna

def objective1(trial):
    x = trial.suggest_float("x", 0, 10)
    y = trial.suggest_float("y", 0, 10)
    return x ** 2 + y ** 2

def objective2(trial):
    x = trial.suggest_float("x", 10, 20)
    y = trial.suggest_float("y", 10, 20)
    return x ** 2 + y ** 2

sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
study = optuna.create_study(sampler=sampler)
study.optimize(objective1, n_trials=10)
study.optimize(objective2, n_trials=1)