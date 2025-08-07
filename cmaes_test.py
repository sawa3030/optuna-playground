import optuna
from optuna.visualization import plot_optimization_history
import optunahub
def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    # y = trial.suggest_int("y", -1, 1)
    return x**2

# bbob = optunahub.load_module("benchmarks/bbob")
# sphere2d = bbob.Problem(function_id=1, dimension=1, instance_id=1)

# storage = optuna.storages.get_storage("sqlite:///cmaes_benchmark.db")
source_study = optuna.create_study()
source_study.optimize(objective, 20)
source_trials = source_study.get_trials(deepcopy=False)

sampler = optuna.samplers.CmaEsSampler(
    # x0={"x": 0.5},
    # sigma0=0.1,
    # seed=42,
    # n_startup_trials=5,
    # independent_sampler=optuna.samplers.RandomSampler(),
    # warn_independent_sampling=False,
    # inc_popsize=True,
    # consider_pruned_trials=True,
    use_separable_cma=True,
    # with_margin=True,
    # lr_adapt=True
    # source_trials=source_trials,
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)

# fig = plot_optimization_history(study)
# fig.write_html("cmaes_optimization_history.html")
# import optuna
# import optunahub

# b

