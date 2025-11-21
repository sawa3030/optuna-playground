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
import time
from optuna.visualization import plot_contour

# always show the warnings
# warnings.simplefilter("always")

# def objective(trial: optuna.Trial) -> float:
#     x = trial.suggest_float("x", -5, 5)
#     y = trial.suggest_float("y", -5, 5)
#     return x**2 + y**2

bbob = optunahub.load_module("benchmarks/bbob")
sphere2d = bbob.Problem(function_id=22, dimension=2, instance_id=1)
storage = optuna.storages.InMemoryStorage()

package_name = "samplers/turbosampler"
module = optunahub.load_local_module(
    package="samplers/turbo",
    registry_root="../optunahub-registry/package/",  # Path to the root of the optunahub-registry.
)

sampler = module.TuRBOSampler(seed=42, n_startup_trials=4, n_trust_region=1)
study = optuna.create_study(
    # study_name="turbo_sampler",
    sampler=sampler,
    # directions=["minimize"],
    directions=sphere2d.directions,
    storage=storage,
    )
study.optimize(sphere2d, n_trials=20)

for i in range(50):
    study.optimize(sphere2d, n_trials=1)
    fig = plot_contour(study)   

    ub_by_tr = sampler._ub_by_tr
    lb_by_tr = sampler._lb_by_tr

    # print(ub_by_tr)

    for tr_id in range(len(ub_by_tr)):
        fig.add_shape(
            type="rect",
            x0=lb_by_tr[tr_id]["x0"],
            y0=lb_by_tr[tr_id]["x1"],
            x1=ub_by_tr[tr_id]["x0"],
            y1=ub_by_tr[tr_id]["x1"],
            line=dict(color="yellow"),
        )

    # get the last trial and plot a red circle
    last_trial = study.trials[-1]
    fig.add_shape(
        type="circle",
        x0=last_trial.params["x0"] - 0.2,
        y0=last_trial.params["x1"] - 0.2,
        x1=last_trial.params["x0"] + 0.2,
        y1=last_trial.params["x1"] + 0.2,
        line=dict(color="red"),
    )   

    fig.write_image(f"frames/frame_{i:03}.png")