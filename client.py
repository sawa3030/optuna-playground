import optuna
import optunahub


# module = optunahub.load_module("samplers/restart_cmaes")

# module = optunahub.load_module(
#     # category is one of [pruners, samplers, visualization].
#     package="samplers/restart_cmaes",
#     repo_owner="sawa3030",
#     ref="fix/restart_cmaes",
# )
# RestartCmaEsSampler = module.RestartCmaEsSampler


# def objective(trial: optuna.Trial) -> float:
#     x = trial.suggest_float("x", -1, 1)
#     y = trial.suggest_int("y", -1, 1)
#     return x**2 + y


# sampler = RestartCmaEsSampler()  # CMA-ES without restart (default)
# sampler = RestartCmaEsSampler(restart_strategy="ipop")  # IPOP-CMA-ES
# sampler = RestartCmaEsSampler(None, None, 1, None, True, None, False, restart_strategy="bipop")  # BIPOP-CMA-ES
# study = optuna.create_study(sampler=sampler)
# study.optimize(objective, n_trials=20)

def objective1(trial: optuna.Trial) -> float:
    x0 = trial.suggest_float("x0", 2, 3)
    x1 = trial.suggest_float("x1", 1e-2, 1e2, log=True)
    return x0 + x1

source_study = optuna.create_study()
source_study.optimize(objective1, 20)

# Should not raise an exception.
sampler = optuna.samplers.CmaEsSampler(source_trials=source_study.trials)
target_study1 = optuna.create_study(sampler=sampler)
target_study1.optimize(objective1, 20)

def objective2(trial: optuna.Trial) -> float:
    x0 = trial.suggest_float("x0", 2, 3)
    x1 = trial.suggest_float("x1", 1e-2, 1e2, log=True)
    x2 = trial.suggest_float("x2", 1e-2, 1e2, log=True)
    return x0 + x1 + x2

# Should raise an exception.
sampler = optuna.samplers.CmaEsSampler(source_trials=source_study.trials)
target_study2 = optuna.create_study(sampler=sampler)
# try:
target_study2.optimize(objective2, 20)
# except Error:
#     print("InvalidStudyError raised as expected.")