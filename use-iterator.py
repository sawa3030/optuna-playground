import argparse

import optuna
import time


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

optuna.storages.JournalStorage(
    optuna.storages.journal.JournalRedisBackend(f"redis://localhost:6379")
)
