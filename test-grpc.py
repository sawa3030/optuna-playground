from optuna.storages import run_grpc_proxy_server
from optuna.storages import get_storage

storage = get_storage("postgresql+psycopg://user:password@localhost:5432/optuna_db")
run_grpc_proxy_server(storage, host="localhost", port=13000)