from optuna.storages import InMemoryStorage
from optuna.storages.journal import JournalFileBackend
# from optuna.storages import GrpcServer
# from optuna.storages._grpc import start_grpc_server
from optuna.storages import run_grpc_proxy_server
import optuna
import tempfile
from optuna.trial import TrialState

# url = "sqlite:///example5.db"
# rdb_storage = optuna.storages.RDBStorage(
#     url,
#     engine_kwargs={"connect_args": {"timeout": 300}},
# )
# storage = (
#     optuna.storages._CachedStorage(rdb_storage)
# )
storage = optuna.storages.RDBStorage(
    url="sqlite:///:memory2:",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)

# storage = InMemoryStorage()
# storage = JournalFileBackend(
#     path="journal_file",
# )
# storage = optuna.storages.JournalStorage(
#     optuna.storages.journal.JournalFileBackend("journal_file1"),
# )

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x**2

study = optuna.create_study(storage=storage, direction="minimize")

study.enqueue_trial({"x": 1})

storage.set_trial_state_values(1, state=TrialState.WAITING, values=[])
print(storage.get_trial(1).values)



# gRPC サーバーの起動（別スレッドまたは別プロセスで実行するのが推奨）
# run_grpc_proxy_server(storage, host="localhost", port=50051)
# grpc_server = GrpcServer(storage=storage, port=50051)
# grpc_server.start()
