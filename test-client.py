import optuna

from optuna.storages import GrpcStorageProxy


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    storage = GrpcStorageProxy(host="localhost", port=13000)
    study = optuna.create_study(
        study_name="grpc_proxy_multinode",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)