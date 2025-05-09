import optuna
import optunahub
import cProfile


module = optunahub.load_module(
    # category is one of [pruners, samplers, visualization].
    package="samplers/restart_cmaes",
    repo_owner="sawa3030",
    ref="in-memory",
)
RestartCmaEsSampler = module.RestartCmaEsSampler


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


# sampler = RestartCmaEsSampler()  # CMA-ES without restart (default)
sampler = RestartCmaEsSampler(restart_strategy="ipop", use_system_attrs=False)  # IPOP-CMA-ES
# sampler = RestartCmaEsSampler(restart_strategy="bipop")  # BIPOP-CMA-ES
study = optuna.create_study(sampler=sampler)
# study.optimize(objective, n_trials=20)
cProfile.run("study.optimize(objective, n_trials=500)", filename="cmaes-in.stats")
