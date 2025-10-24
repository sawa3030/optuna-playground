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
module = optunahub.load_local_module(
    package="samplers/turbo",
    registry_root="../optunahub-registry/package/",  # Path to the root of the optunahub-registry.
)
# sampler = optuna.samplers.GPSampler(seed=1)

turbo_best_values = []
gp_best_values = []

for i in range(10):
    sampler = module.TuRBOSampler(seed=i, failure_tolerance=10)
    study1 = optuna.create_study(
        # study_name="turbo_sampler",
        sampler=sampler,
        directions=sphere2d.directions,
        storage=storage,
        )
    study1.optimize(sphere2d, n_trials=200)
    turbo_best_values.append(study1.best_value)

for i in range(10):
    study2 = optuna.create_study(
        # study_name="gp_sampler",
        sampler=optuna.samplers.GPSampler(seed=i),
        directions=sphere2d.directions,
        storage=storage,
    )
    study2.optimize(sphere2d, n_trials=200)
    gp_best_values.append(study2.best_value)

print("TuRBO Sampler best values:", turbo_best_values)
print("GP Sampler best values:", gp_best_values)

# mean and std of best values
import numpy as np
print("TuRBO Sampler mean best value:", np.mean(turbo_best_values))
print("TuRBO Sampler std of best values:", np.std(turbo_best_values))
print("GP Sampler mean best value:", np.mean(gp_best_values))
print("GP Sampler std of best values:", np.std(gp_best_values))


# fig = optuna.visualization.plot_optimization_history([study1, study2])
# fig.write_html("history_turbo.html", auto_open=True)
# fig.write_html("history_gp.html", auto_open=True)