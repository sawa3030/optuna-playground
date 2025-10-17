from pathlib import Path

import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


# storage = optuna.storages.JournalStorage(
#     optuna.storages.journal.JournalFileBackend("./journal_storage.log")
# )
# sampler = optuna.samplers.RandomSampler()
# study = optuna.create_study(storage=storage, sampler=sampler)
# study.optimize(objective, n_trials=100000)

for num in [1 << i for i in range(1, 20)]:
    Path(f"journal_storage{num}.log").write_text(
    "\n".join(Path("journal_storage.log").read_text().splitlines()[:num])
    )