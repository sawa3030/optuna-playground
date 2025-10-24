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
# warnings.simplefilter("always")

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2

bbob = optunahub.load_module("benchmarks/bbob")
sphere2d = bbob.Problem(function_id=22, dimension=10, instance_id=1)
storage = optuna.storages.InMemoryStorage()

package_name = "samplers/turbosampler"
sampler = optunahub.load_local_module(
    package="samplers/turbo",
    registry_root="../optunahub-registry/package/",  # Path to the root of the optunahub-registry.
).TuRBOSampler(seed=1, failure_tolerance=10)
# sampler = optuna.samplers.GPSampler(seed=1)

study1 = optuna.create_study(
    study_name="turbo_sampler",
    sampler=sampler,
    directions=sphere2d.directions,
    storage=storage,
    )
study1.optimize(sphere2d, n_trials=500)

study2  = optuna.create_study(
    study_name="gp_sampler",
    sampler=optuna.samplers.GPSampler(seed=1),
    directions=sphere2d.directions,
    storage=storage,
    )
study2.optimize(sphere2d, n_trials=500)

fig = optuna.visualization.plot_optimization_history([study1, study2])
fig.write_html("history_turbo.html", auto_open=True)
# fig.write_html("history_gp.html", auto_open=True)