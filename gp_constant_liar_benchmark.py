import optuna
import math
from plotly.io import show
import optunahub
import cocoex as ex

# def objective(trial):
#     x1 = trial.suggest_float("x1", -5, 10)
#     x2 = trial.suggest_float("x2", 0, 15)
#     return x1**2 + x2

# def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
#     c1 = 10 - trial.params["x1"] - trial.params["x2"]
#     c2 = trial.params["x1"] - 5
#     return (c1, c2)

# study = optuna.create_study(sampler=optuna.samplers.GPSampler(constant_liar=True, constraints_func=constraints), directions=["minimize", "maximize"])
# study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=42, constant_liar=True))
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(constant_liar=True))
# study.optimize(objective, n_trials=20)

a = 1
b = 5.1 / (4 * math.pi * math.pi)
c = 5 / math.pi
r = 6
s = 10
t = 1 / (8 * math.pi)

problem=ex.Suite("bbob", "", "").get_problem_by_function_dimension_instance(
    function=22,
    dimension=2, 
    instance=1
)

def objective(trial):
    x1 = trial.suggest_float("x1", -5, 10)
    x2 = trial.suggest_float("x2", 0, 15)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*math.cos(x1) + s

repeat_times = 1
batch_size = 10
batch_repeats = 5
objective_value_worst = [[] for _ in range(repeat_times)]
objective_value_best = [[] for _ in range(repeat_times)]
objective_value_average = [[] for _ in range(repeat_times)]

objective_value_worst_best = [[] for _ in range(repeat_times)]
objective_value_best_best = [[] for _ in range(repeat_times)]
objective_value_average_best = [[] for _ in range(repeat_times)]

for seed in range(repeat_times):
    study_worst = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=seed, constant_liar="average"), study_name="constant_liar_worst", directions=["minimize"])
    # study_worst.optimize(objective, n_trials=10)
    for j in range(batch_repeats):
        trials = []
        x1s = []
        x2s = []
        for i in range(batch_size):
            trial = study_worst.ask()
            x1 = trial.suggest_float("x1", -5, 10)
            x2 = trial.suggest_float("x2", 0, 15)
            # x1 = trial.suggest_float("x1", problem.lower_bounds[0], problem.upper_bounds[0])
            # x2 = trial.suggest_float("x2", problem.lower_bounds[1], problem.upper_bounds[1])
            trials.append(trial)
            x1s.append(x1)
            x2s.append(x2)
            print(f"Trial {i}: x1={x1}, x2={x2}")

        for i in range (batch_size):
            objective_value = a*(x2s[i] - b*x1s[i]**2 + c*x1s[i] - r)**2 + s*(1 - t)*math.cos(x1s[i]) + s
            # objective_value = problem([x1s[i], x2s[i]])
            study_worst.tell(trials[i], [objective_value])
            print(f"Trial {i}: objective_value={objective_value}")
            objective_value_worst[seed].append(objective_value)
            objective_value_worst_best[seed].append(study_worst.best_value)
    fig = optuna.visualization.plot_contour(study_worst, params=["x1", "x2"], target_name="Objective Value")
    fig.write_image("constant_liar_mean.png")  

    # study_best = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=seed, constant_liar="best"), study_name="constant_liar_best", directions=["minimize"])
    # # study_best.optimize(objective, n_trials=10)
    # for j in range(batch_repeats):
    #     trials = []
    #     x1s = []
    #     x2s = []
    #     for i in range(batch_size):
    #         trial = study_best.ask()
    #         # x1 = trial.suggest_float("x1", -5, 10)
    #         # x2 = trial.suggest_float("x2", 0, 15)
    #         x1 = trial.suggest_float("x1", problem.lower_bounds[0], problem.upper_bounds[0])
    #         x2 = trial.suggest_float("x2", problem.lower_bounds[1], problem.upper_bounds[1])
    #         trials.append(trial)
    #         x1s.append(x1)
    #         x2s.append(x2)
    #         print(f"Trial {i}: x1={x1}, x2={x2}")

    #     for i in range (batch_size):
    #         # objective_value = a*(x2s[i] - b*x1s[i]**2 + c*x1s[i] - r)**2 + s*(1 - t)*math.cos(x1s[i]) + s
    #         objective_value = problem([x1s[i], x2s[i]])
    #         study_best.tell(trials[i], [objective_value])
    #         print(f"Trial {i}: objective_value={objective_value}")
    #         objective_value_best[seed].append(objective_value)
    #         objective_value_best_best[seed].append(study_best.best_value)

    # study_average = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=seed, constant_liar="average"), study_name="constant_liar_average", directions=["minimize"])
    # # study_average.optimize(objective, n_trials=10)
    # for j in range(batch_repeats):
    #     trials = []
    #     x1s = []
    #     x2s = []
    #     for i in range(batch_size):
    #         trial = study_average.ask()
    #         # x1 = trial.suggest_float("x1", -5, 10)
    #         # x2 = trial.suggest_float("x2", 0, 15)
    #         x1 = trial.suggest_float("x1", problem.lower_bounds[0], problem.upper_bounds[0])
    #         x2 = trial.suggest_float("x2", problem.lower_bounds[1], problem.upper_bounds[1])
    #         trials.append(trial)
    #         x1s.append(x1)
    #         x2s.append(x2)
    #         print(f"Trial {i}: x1={x1}, x2={x2}")

    #     for i in range (batch_size):
    #         # objective_value = a*(x2s[i] - b*x1s[i]**2 + c*x1s[i] - r)**2 + s*(1 - t)*math.cos(x1s[i]) + s
    #         objective_value = problem([x1s[i], x2s[i]])
    #         study_average.tell(trials[i], [objective_value])
    #         print(f"Trial {i}: objective_value={objective_value}")
    #         objective_value_average[seed].append(objective_value)
    #         objective_value_average_best[seed].append(study_average.best_value)

# get the average and plot the optimization history
# average_worst = [sum(x)/len(x) for x in zip(*objective_value_worst)]
# average_best = [sum(x)/len(x) for x in zip(*objective_value_best)]
# average_average = [sum(x)/len(x) for x in zip(*objective_value_average)] 
# average_worst_best = [sum(x)/len(x) for x in zip(*objective_value_worst_best)]
# average_best_best = [sum(x)/len(x) for x in zip(*objective_value_best_best)]
# average_average_best = [sum(x)/len(x) for x in zip(*objective_value_average_best)]

# import plotly.graph_objects as go
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=average_worst, mode='markers', name='Worst'))
# fig.add_trace(go.Scatter(y=average_best, mode='markers', name='Best'))
# fig.add_trace(go.Scatter(y=average_average, mode='markers', name='Average'))

# fig.add_trace(go.Scatter(y=average_worst_best, mode='lines', name='Worst Best Value'))
# fig.add_trace(go.Scatter(y=average_best_best, mode='lines', name='Best Best Value'))
# fig.add_trace(go.Scatter(y=average_average_best, mode='lines', name='Average Best Value'))

# fig.update_layout(title='GP Constant Liar Strategy Comparison',
#                    xaxis_title='Trial',
#                    yaxis_title='Objective Value')
# fig.write_image("gp_constant_liar_comparison.png")  

# fig = optuna.visualization.plot_optimization_history([study_worst, study_best, study_average], target_name="Objective Value")
# fig.write_image(f"gp_constant_liar_mean.png")

# def objective1(trial):
#     a1 = trial.suggest_float("a1", -10, 10)
#     a2 = trial.suggest_float("a2", -10, 10)
#     return a1 + a2 / 1000
# study.optimize(objective1, n_trials=20)
