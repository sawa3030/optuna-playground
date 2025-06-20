import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial
import threading


def objective(trial: Trial):
    print(f"Running trial {trial.number=} in {threading.current_thread().name}")
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study(
    # study_name="journal_storage_multithread",
    storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
    # load_if_exists=True,
)
study.optimize(objective, n_trials=20, n_jobs=4)