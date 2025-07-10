import optuna
import warnings

warnings.simplefilter("always")

# def objective(trial):
#     x = trial.suggest_float("x", -1, 1)
#     y = trial.suggest_int("y", -1, 1)
#     return x**2 + y, x



# sampler = optuna.samplers.TPESampler(
#     seed=42,
#     multivariate=True,
#     constant_liar=True,
# )
# study = optuna.create_study(directions=["minimize", "minimize"],sampler=sampler)
# study.optimize(objective, n_trials=20)
# print(sampler._search_space._search_space)


import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

N_TRIAL = 50
N_BATCH = 10
# N_TRIAL = 5
# N_BATCH = 2

# for multivariate in (True):
    # for constant_liar in (True):
multivariate = True
constant_liar = True
sampler = optuna.samplers.TPESampler(
    seed=42,
    multivariate=multivariate,
    constant_liar=constant_liar,
)
study = optuna.create_study(sampler=sampler)

for i in range(0, N_TRIAL, N_BATCH):
    trials = []
    for j in range(N_BATCH):
        trials.append(study.ask())
    # X = [trial.suggest_float("x", -10, 10) for trial in trials]
    # Y = [trial.suggest_float("y", -10, 10) for trial in trials]
    
    for trial in trials:


        states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        trials_temp = study._get_trials(deepcopy=False, states=states)
        for t in trials_temp:
            pass
            # print("trial.params", t.params)
            # print("sampler._get_params", sampler._get_params(t))
        X = trial.suggest_float("x", -10, 10)
        Y = trial.suggest_float("y", -10, 10)

    # for j in range(N_BATCH):
    #     study.tell(trials[j], X[j] ** 2 + Y[j] ** 2)

            # Skip first random sampling.
        #     if i > 0:
        #         plt.plot(X, Y, ".")
        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)
        # plt.savefig(f"{multivariate}-{constant_liar}.png")
        # plt.clf()