# import optuna
# from optuna.trial import TrialState
# import optunahub

# storage = optuna.storages.RDBStorage(
#     url="sqlite:///:memory:",
#     engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
# )

# def objective(trial: optuna.Trial) -> float:
#     x = trial.suggest_float("x", -1, 1)
#     y = trial.suggest_int("y", -1, 1)
#     if y == 0:
#         trial.storage.set_trial_state_values(
#             trial._trial_id, TrialState.COMPLETE, [0.0]
#         )
#     return x**2 + y

# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), storage=storage)
# study.optimize(objective, n_trials=10)


import os

from optuna.storages import run_grpc_proxy_server
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalStorage


try:
    os.remove("test-grpc.log")
except FileNotFoundError:
    pass
storage = JournalStorage(JournalFileBackend("test-grpc.log"))
run_grpc_proxy_server(storage, host="localhost", port=13000)

