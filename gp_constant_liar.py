import optuna
from optuna.importance import PedAnovaImportanceEvaluator
import warnings
import cProfile
import optunahub

warnings.simplefilter("always")

def objective(trial):
    x1 = trial.suggest_float("x1", -10, 10)
    x2 = trial.suggest_float("x2", -10, 10)
    return x1**2 + x2

def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    c1 = 10 - trial.params["x1"] - trial.params["x2"]
    c2 = trial.params["x1"] - 5
    return (c1, c2)

wfg = optunahub.load_module("benchmarks/wfg")
wfg4 = wfg.Problem(function_id=4, n_objectives=2, dimension=3, k=1)

# study = optuna.create_study(sampler=optuna.samplers.GPSampler(constant_liar=True, constraints_func=constraints), directions=["minimize", "maximize"])
# study = optuna.create_study(sampler=optuna.samplers.GPSampler(constant_liar="worst"))
study = optuna.create_study(sampler=optuna.samplers.GPSampler(n_startup_trials = 10), directions=wfg4.directions)
# study = optuna.create_study(sampler=optuna.samplers.GPSampler())
# study = optuna.create_study(sampler=optuna.samplers.TPESampler(constant_liar=True))
study.optimize(wfg4, n_trials=10)
# cProfile.run("study.optimize(objective, n_trials=1)", filename="output.stats")


for j in range(8):
    trials = []
    x0s = []
    x1s = []
    x2s = []
    for i in range(5):
        trial = study.ask()
        trials.append(trial)

    for i in range(5):
        # print(study.trials[trials[i].number])
        # x1 = trials[i].suggest_float("x1", -10, 10)
        # x2 = trials[i].suggest_float("x2", -10, 10)
        # print(f"Trial {i}: x1={x1}, x2={x2}")
        # print(wfg4.search_space)
        x0 = trials[i].suggest_float("x0", 0.0, 2.0)
        x1 = trials[i].suggest_float("x1", 0.0, 4.0)
        x2 = trials[i].suggest_float("x2", 0.0, 6.0)

    for i in range (5):
        study.tell(trials[i], wfg4.evaluate(trials[i].params))
        print(trials[i].params, wfg4.evaluate(trials[i].params))

    print("===")

# for j in range(1):
#     trials = []
#     for i in range(5):
#         trial = study.ask()
#         trials.append(trial)

#     for i in range(5):
#         # print(study.trials[trials[i].number])
#         x1 = trials[i].suggest_float("x1", -10, 10)
#         # x2 = trials[i].suggest_float("x2", -10, 10)
#         print(f"Trial {i}: x1={x1}")

#     for i in range(5):
#         # print(study.trials[trials[i].number])
#         # x1 = trials[i].suggest_float("x1", -10, 10)
#         x2 = trials[i].suggest_float("x2", -10, 10)
#         print(f"Trial {i}: x2={x2}")

#     for i in range (5):
#         study.tell(trials[i], [10])

# def objective1(trial):
#     a1 = trial.suggest_float("a1", -10, 10)
#     a2 = trial.suggest_float("a2", -10, 10)
#     return a1 + a2 / 1000
# study.optimize(objective1, n_trials=20)

# study_pareto = optuna.create_study(
#     study_name="ParetoFront", directions=wfg4.directions
# )
# for x in wfg4.get_optimal_solutions(1000):  # Generate 1000 Pareto optimal solutions
#     study_pareto.enqueue_trial(params={
#         f"x{i}": x.phenome[i] for i in range(3)
#     })
# study_pareto.optimize(wfg4, n_trials=10)

fig = optuna.visualization.plot_pareto_front(study)
# optunahub.load_module("visualization/plot_pareto_front_multi").plot_pareto_front(
#     [study_pareto, study]
# ).write_image("pareto_front.png")
fig.write_image("pareto_front_none.png")
