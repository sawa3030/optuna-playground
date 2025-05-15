# import optuna
# import cProfile
# import sys
# import pstats

# def objective(trial: optuna.Trial) -> float:
#     x = trial.suggest_float("x", -1, 1)
#     y = trial.suggest_int("y", -1, 1)
#     return x**2 + y


# study = optuna.create_study()
# cProfile.run("study.optimize(objective, n_trials=1000)", filename="pr6079.stats")
# # cProfile.run("study.optimize(objective, n_trials=1000)", filename="master.stats")

import warnings

# 全てのWarningを表示
warnings.simplefilter("always")
import optuna
import logging

class ListHandler(logging.Handler):
    def emit(self, record):
        log_capture.append(self.format(record))

_logger = logging.getLogger("optuna")
log_capture = []
handler = ListHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.WARNING)

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


study = optuna.create_study()
frozen_trial = optuna.study._optimize._run_trial(study, lambda _: float("nan"), catch=())
for log in log_capture:
    print(f"Caught log: {log}")
# study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
# study.optimize(objective, n_trials=10000)