import optuna
import optunahub
import cProfile

from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
# import sys
# import pstats
# import time

# args = sys.argv
# dir_name = args[1]
# file_name = args[2]

# storage = optuna.storages.RDBStorage(
#     url="sqlite:///:memory:",
#     engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
# )
# if dir_name == 'master_5000':
#     module = optunahub.load_module(
#         package="samplers/restart_cmaes",
#     )
# else:
module = optunahub.load_module(
    package="samplers/restart_cmaes",
    repo_owner="sawa3030",
    ref="fix/use-in-memory-in-restartcmaessampler-1",
)
RestartCmaEsSampler = module.RestartCmaEsSampler


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


# sampler = RestartCmaEsSampler()  # CMA-ES without restart (default)
sampler = RestartCmaEsSampler(restart_strategy="ipop", seed = 42, store_optimizer_state_in_storage = False)  # IPOP-CMA-ES
# sampler = RestartCmaEsSampler(restart_strategy="bipop")  # BIPOP-CMA-ES
# study = optuna.create_study(sampler=sampler)
# start = time.time()
def run_optimization(_):
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
        load_if_exists=True,
        sampler=sampler)
    study.optimize(objective, n_trials=10)
    print(len(sampler._optimizer_states_by_trial))
if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(run_optimization, range(3))
# end = time.time()
# print(f"Total time taken: {end - start:.2f} seconds")
# profiller = cProfile.Profile()
# profiller.run("study.optimize(objective, n_trials=200)")

# profiller.dump_stats('./'+dir_name+'/'+file_name+'.prof')
# stats = pstats.Stats(profiller)
# function_name = "optimize"
# function_name_1 = "_tell_with_warning"
# i = 0
# cumtime_global = 0

# sys.stdout = open('./'+dir_name+'/'+file_name+'.txt', 'w')
# for func, data in stats.stats.items():
#     if function_name in func[2]:
#         ncalls, tottime, percall, cumtime, percall_cum = data
#         print(f"関数: {func[2]}")
#         print(f"  呼び出し回数: {ncalls}")
#         print(f"  合計実行時間 (tottime): {tottime:.6f} 秒")
#         print(f"  累積時間 (cumtime): {cumtime:.6f} 秒")
#         cumtime_global += cumtime
#         i += 1
#         if i == 2:
#             break

# sys.stdout.close()

# f = open('./'+dir_name+'/'+'sum.txt', 'a')
# f.write(str(end - start) + '\n')
# f.close()