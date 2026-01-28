
# import optuna
# import optunahub


# def objective(trial: optuna.trial.Trial) -> float:
#     x = trial.suggest_float("x", -10, 10)
#     y = trial.suggest_int("y", -10, 10)
#     return x**2 + y**2


# module = optunahub.load_local_module(
#     package="samplers/hebo",
#     registry_root="../optunahub-registry/package/"
#     )
# sampler = module.HEBOSampler()
# study = optuna.create_study(sampler=sampler)
# study.optimize(objective, n_trials=20)

# print(study.best_trial.params, study.best_trial.value)
# params = study.best_trial.params

# #print params type
# print(type(params["x"]), type(params["y"]))

import optuna
import optunahub


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    z = trial.suggest_categorical("z", ["a", "b", "c"])
    print(type(x))
    return x**2 + y**2


module = optunahub.load_module(
  package = "samplers/hebo",
  repo_owner = 'sawa3030',
  repo_name = 'optunahub-registry',
  ref = 'main',
  force_reload=True)
sampler = module.HEBOSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

print(study.best_trial.params, study.best_trial.value)