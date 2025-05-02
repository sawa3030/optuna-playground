import optuna
import optunahub
from optuna.visualization import plot_optimization_history
from plotly.io import show
 
bbob = optunahub.load_module("benchmarks/bbob")
sphere2d = bbob.Problem(function_id=1, dimension=2, instance_id=1)

storage = optuna.storages.get_storage("sqlite:///cmaes_benchmark.db")

sampler = optuna.samplers.CmaEsSampler(seed=1, restart_strategy="ipop")
study_v4_3_0 = optuna.create_study(
    sampler=sampler,
    directions=sphere2d.directions,
    study_name="cmaes_v4_3_0",
    storage=storage
)
study_v4_3_0.optimize(sphere2d, n_trials=500)


module = optunahub.load_module(
    package="samplers/restart_cmaes",
    repo_owner="sawa3030",
    ref="remove-options",
)
RestartCmaEsSampler = module.RestartCmaEsSampler
sampler = RestartCmaEsSampler(seed=1, restart_strategy="ipop")
study_pr266 = optuna.create_study(
    sampler=sampler,
    directions=sphere2d.directions,
    study_name="cmaes_pr266",
    storage=storage
)
study_pr266.optimize(sphere2d, n_trials=500)

fig = plot_optimization_history([study_v4_3_0, study_pr266])
fig.write_html("history_ipop.html", auto_open=True)