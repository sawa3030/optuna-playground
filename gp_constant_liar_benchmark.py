import optuna
import math

a = 1
b = 5.1 / (4 * math.pi * math.pi)
c = 5 / math.pi
r = 6
s = 10
t = 1 / (8 * math.pi)

for constant_liar in [None, "worst", "best", "mean"]:
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=1, constant_liar=constant_liar), directions=["minimize"])
    for j in range(5):
        trials = []
        x1s = []
        x2s = []
        for i in range(10):
            trial = study.ask()
            x1 = trial.suggest_float("x1", -5, 10)
            x2 = trial.suggest_float("x2", 0, 15)
            trials.append(trial)
            x1s.append(x1)
            x2s.append(x2)
            print(f"Trial {i}: x1={x1}, x2={x2}")

        for i in range (10):
            # Branin-Hoo function is used as objective function
            objective_value = a*(x2s[i] - b*x1s[i]**2 + c*x1s[i] - r)**2 + s*(1 - t)*math.cos(x1s[i]) + s
            study.tell(trials[i], [objective_value])
            print(f"Trial {i}: objective_value={objective_value}")
    fig = optuna.visualization.plot_contour(study, params=["x1", "x2"])
    fig.write_image(f"constant_liar_{constant_liar}.png")  

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
