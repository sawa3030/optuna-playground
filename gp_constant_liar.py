import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import warnings
import cProfile

warnings.simplefilter("always")

def objective(trial):
    x1 = trial.suggest_float("x1", -10, 10)
    x2 = trial.suggest_float("x2", -10, 10)
    return x1**2 + x2

def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    c1 = 10 - trial.params["x1"] - trial.params["x2"]
    c2 = trial.params["x1"] - 5
    return (c1, c2)

# study = optuna.create_study(sampler=optuna.samplers.GPSampler(constant_liar=True, constraints_func=constraints), directions=["minimize", "maximize"])
study = optuna.create_study(sampler=optuna.samplers.GPSampler(constant_liar="worst"))
# study = optuna.create_study(sampler=optuna.samplers.GPSampler())
# study = optuna.create_study(sampler=optuna.samplers.GPSampler())
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(constant_liar=True))
study.optimize(objective, n_trials=20)
# cProfile.run("study.optimize(objective, n_trials=1)", filename="output.stats")


for j in range(3):
    trials = []
    for i in range(5):
        trial = study.ask()
        trials.append(trial)

    for i in range(5):
        print(study.trials[trials[i].number])
        x1 = trials[i].suggest_float("x1", -10, 10)
        x2 = trials[i].suggest_float("x2", -10, 10)
        print(f"Trial {i}: x1={x1}, x2={x2}")

    for i in range (5):
        study.tell(trials[i], [10])

for j in range(1):
    trials = []
    for i in range(5):
        trial = study.ask()
        trials.append(trial)

    for i in range(5):
        # print(study.trials[trials[i].number])
        x1 = trials[i].suggest_float("x1", -10, 10)
        # x2 = trials[i].suggest_float("x2", -10, 10)
        print(f"Trial {i}: x1={x1}")

    for i in range(5):
        # print(study.trials[trials[i].number])
        # x1 = trials[i].suggest_float("x1", -10, 10)
        x2 = trials[i].suggest_float("x2", -10, 10)
        print(f"Trial {i}: x2={x2}")

    for i in range (5):
        study.tell(trials[i], [10])

# def objective1(trial):
#     a1 = trial.suggest_float("a1", -10, 10)
#     a2 = trial.suggest_float("a2", -10, 10)
#     return a1 + a2 / 1000
# study.optimize(objective1, n_trials=20)
