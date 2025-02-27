import optuna
import cProfile
import time

# def objective(trial): 
#     x = trial.suggest_uniform('x', -10, 10)
#     return (x - 2) ** 2

study = optuna.create_study()

# study.enqueue_trial(params={"x": -5, "y": 5})
# study.enqueue_trial(params={"x": -1, "y": 0})

def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    y = trial.suggest_uniform("y", -10, 10)
    return x**2 + y**2

study.optimize(objective, n_trials=100)

for i in range(-5, 5):
    study.enqueue_trial({"x": i, "y": i})

def profile_objective():
    study.optimize(objective, n_trials=100)

cProfile.run('profile_objective()', 'journal_storage_master_100.prof')
# cProfile.run('profile_objective()', 'journal_storage_fix_100.prof')

