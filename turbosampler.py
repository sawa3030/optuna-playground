"""
This example is only for sampler.
You can verify your sampler code using this file as well.
Please feel free to remove this file if necessary.
"""

from __future__ import annotations

import optuna
import optunahub
import warnings
from plotly.io import show

# always show the warnings
warnings.simplefilter("always")

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2

bbob = optunahub.load_module("benchmarks/bbob")
sphere2d = bbob.Problem(function_id=1, dimension=2, instance_id=1)
# storage = optuna.storages.get_storage("sqlite:///cmaes_benchmark.db")
storage = optuna.storages.InMemoryStorage()

package_name = "samplers/turbosampler"
sampler = optunahub.load_local_module(
    package="samplers/turbo",
    registry_root="../optunahub-registry/package/",  # Path to the root of the optunahub-registry.
).TuRBOSampler(seed=0)

study = optuna.create_study(
    sampler=sampler,
    # directions=sphere2d.directions,
    # study_name="turbo_study",
    storage=storage,
    # load_if_exists=False,
    )
study.optimize(objective, n_trials=100)
print(study.best_trials)

fig = optuna.visualization.plot_optimization_history(study)
fig.write_html("history_turbo.html", auto_open=True)