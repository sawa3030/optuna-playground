import optuna
import cProfile



def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


study = optuna.create_study()
cProfile.run("study.optimize(objective, n_trials=500)", filename=".stats")
