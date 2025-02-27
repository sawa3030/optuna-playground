import optuna
import cProfile
import time
import sys
import pstats

args = sys.argv

num_of_waiting_trials = int(args[1])
num_of_trials = int(args[2])
dir_name = args[3]
file_name = args[4]

study = optuna.create_study()

def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return x**2

for i in range(-num_of_waiting_trials // 2, num_of_waiting_trials // 2):
    study.enqueue_trial({"x": i})

def profile_objective():
    study.optimize(objective, n_trials=num_of_trials)

profiller = cProfile.Profile()
profiller.run('profile_objective()')

profiller.dump_stats('./'+dir_name+'/'+file_name+'.prof')

# sys.stdout = open('./'+dir_name+'/'+file_name+'.txt', 'w')
# profiller.print_stats('_pop_waiting_trial_id|get_all_trials')

stats = pstats.Stats(profiller)
function_name = "_pop_waiting_trial_id"
function_name_1 = "get_all_trials"
i = 0

sys.stdout = open('./'+dir_name+'/'+file_name+'.txt', 'w')
for func, data in stats.stats.items():
    if function_name in func[2] or function_name_1 in func[2]:  # 関数名が一致するものを探す
        ncalls, tottime, percall, cumtime, percall_cum = data
        print(f"関数: {func[2]}")
        print(f"  呼び出し回数: {ncalls}")
        print(f"  合計実行時間 (tottime): {tottime:.6f} 秒")
        print(f"  累積時間 (cumtime): {cumtime:.6f} 秒")
        i += 1
        if i == 2:
            break
