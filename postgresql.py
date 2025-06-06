import optuna
from optuna.storages import RDBStorage
from sqlalchemy import NullPool


def main():
    study_name = "study"
    optuna_database_url = "postgresql+psycopg://user:password@localhost:5432/optuna_db"

    print(f"Attempting to create study: {study_name} in {optuna_database_url}")

    storage = RDBStorage(
        url=optuna_database_url,

        engine_kwargs={"poolclass": NullPool},
    )

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="maximize")

    print(f"Study created. ID: {study._study_id}. Exiting to check schema.")

    # Add a dummy trial to trigger the error later
    def objective(trial):
        return trial.suggest_float("x", 0, 1)

    study.optimize(objective, n_trials=1)
    print(f"Best trial: {study.best_trial.value}")  # This will trigger the error


if __name__ == "__main__":
    main()
