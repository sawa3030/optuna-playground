# Online Python Playground
# Use the online IDE to write, edit & run your Python code
# Create, edit & delete files online

print("Try programiz.pro")

import numpy as np

values = np.array([-1, -np.inf, 0, np.inf, 1])
values = np.array([-1])

finite_vals = values[np.isfinite(values)]
print(finite_vals)
best_finite_val = np.max(finite_vals, axis=0, initial=0.0)
worst_finite_val = np.min(finite_vals, axis=0, initial=0.0)
print(best_finite_val)
print(np.max([1], initial=10, axis=0))
print("============")

is_values_finite = np.isfinite(values)
print("======== is_values_finite")
print(is_values_finite)
if np.all(is_values_finite):
    print(values)

print(np.min(np.where(is_values_finite, values, np.inf), axis=0))

is_any_finite = np.any(is_values_finite, axis=0)
# NOTE(nabenabe): values cannot include nan to apply np.clip properly, but Optuna anyways won't
# pass nan in values by design.
ans = np.clip(
    values,
    np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
    np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
)

# finite_vals = values[np.isfinite(values)]
# best_finite_val = np.max(finite_vals, axis=0, initial=0.0)
# worst_finite_val = np.min(finite_vals, axis=0, initial=0.0)

# ans = np.clip(values, worst_finite_val, best_finite_val)

print(ans)

# import optuna
# import cProfile
# import time
# import sys
# import pstats
# from collections.abc import Sequence

# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     return (x - 2) ** 2

# def constraints_func(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
#     return (float('inf') + trial.number, 2)

# def _create_new_trial(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
#     trial_id = study._storage.create_new_trial(study._study_id)
#     return study._storage.get_trial(trial_id)

# # study = optuna.create_study()
# dist = optuna.distributions.FloatDistribution(1.0, 10.0)
# # trial = optuna.trial.FrozenTrial(
# #     number=0,
# #     value=10.0,
# #     state=optuna.trial.TrialState.COMPLETE,
# #     datetime_start=None,
# #     datetime_complete=None,
# #     params={"param-a": 10.0},
# #     distributions={"param-a": dist},
# #     user_attrs={},
# #     system_attrs={},
# #     intermediate_values={},
# #     trial_id=1,
# # )
# # trial = _create_new_trial(study)
# sampler = optuna.samplers.GPSampler(seed=42, constraints_func=constraints_func)
# study = optuna.create_study(sampler=sampler)
# # suggestion = sampler.sample_relative(study, trial, {"param-a": dist})

# # print("Suggestion:")
# # print(suggestion)
# study.optimize(objective, n_trials=10)
