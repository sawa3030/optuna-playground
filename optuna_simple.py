import optuna
import cProfile
import time
import sys
import pstats

# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     return (x - 2) ** 2

# study = optuna.create_study()
# trial = study.ask()
# trial.report(0.5, step=1)

# # 試行を COMPLETE にする
# study.tell(trial, 0.1)

# # 変更を試みる → エラー発生
# trial.report(0.3, step=2)


storage = optuna.storages.RDBStorage(
    url="sqlite:///:memory:",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)
study = optuna.create_study(storage = storage)

def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return x**2

study.optimize(objective, n_trials=2)
try:
    storage.set_trial_intermediate_value(1, 1, 100)
except RuntimeError as e:
    print("catches the error")
    print(e)

# print(study._study_id)

# trials = storage.get_all_trials(1, deepcopy=False)
# print(trials[0])
# trials[0].state = optuna.trial._state.TrialState.WAITING
# print("=====================")
# print(trials[0])

# trials = study.get_trials(deepcopy=False)
# print(trials[0])
# trials[0].state = optuna.trial._state.TrialState.WAITING
# print("=====================")
# print(trials[0])

# # frozen_trial = study.best_trial
# # print(frozen_trial)
# # print(type(frozen_trial))

# # frozen_trial.state = optuna.trial._state.TrialState.WAITING
# # print("=====================")
# # print(frozen_trial)
