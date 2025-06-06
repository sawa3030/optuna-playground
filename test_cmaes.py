import optuna
import optunahub
import cProfile

import sys
import pstats

args = sys.argv
dir_name = args[1]
file_name = args[2]

storage = optuna.storages.RDBStorage(
    url="sqlite:///:memory:",
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)
if dir_name == 'master_5000':
    module = optunahub.load_module(
        package="samplers/restart_cmaes",
    )
else:
    module = optunahub.load_module(
        package="samplers/restart_cmaes",
        repo_owner="sawa3030",
        ref="in-memory",
    )
RestartCmaEsSampler = module.RestartCmaEsSampler


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


# sampler = RestartCmaEsSampler()  # CMA-ES without restart (default)
sampler = RestartCmaEsSampler(restart_strategy="ipop")  # IPOP-CMA-ES
# sampler = RestartCmaEsSampler(restart_strategy="bipop")  # BIPOP-CMA-ES
study = optuna.create_study(sampler=sampler, storage=storage)
# study.optimize(objective, n_trials=20)
profiller = cProfile.Profile()
profiller.run("study.optimize(objective, n_trials=200)")

profiller.dump_stats('./'+dir_name+'/'+file_name+'.prof')
stats = pstats.Stats(profiller)
function_name = "sample_relative"
# function_name_1 = "_tell_with_warning"
i = 0
cumtime_global = 0

sys.stdout = open('./'+dir_name+'/'+file_name+'.txt', 'w')
for func, data in stats.stats.items():
    if function_name in func[2]:
        ncalls, tottime, percall, cumtime, percall_cum = data
        print(f"関数: {func[2]}")
        print(f"  呼び出し回数: {ncalls}")
        print(f"  合計実行時間 (tottime): {tottime:.6f} 秒")
        print(f"  累積時間 (cumtime): {cumtime:.6f} 秒")
        cumtime_global += cumtime
        i += 1
        if i == 2:
            break

sys.stdout.close()

f = open('./'+dir_name+'/'+'sum.txt', 'a')
f.write(str(cumtime_global))
f.write('\n')
f.close()