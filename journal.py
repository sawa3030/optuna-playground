# import optuna
# from optuna.storages import JournalStorage
# from optuna.storages.journal import JournalFileBackend
# from optuna.trial import Trial
# import threading
# import warnings


# warnings.simplefilter("always")

# def objective(trial: Trial):
#     print(f"Running trial {trial.number=} in {threading.current_thread().name}")
#     x = trial.suggest_float("x", -10, 10)
#     return (x - 2) ** 2


# study = optuna.create_study(
#     storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
# )
# study.optimize(objective, n_trials=500, n_jobs=100)


import optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
import warnings

warnings.simplefilter("always")

def objective(trial):
    print(f"Running trial {trial.number=} in process {os.getpid()}")
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def run_optimization(_):
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
        load_if_exists=True, # Useful for multi-process or multi-node optimization.
    )
    study.optimize(objective, n_trials=3)

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(run_optimization, range(12))