import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import warnings

warnings.simplefilter("always")

N_TRIAL = 50
N_BATCH = 10

# for multivariate in (False, True):
#     for constant_liar in (False, True):
sampler = optuna.samplers.TPESampler(
    seed=42,
    multivariate=True,
    constant_liar=True,
)
study = optuna.create_study(sampler=sampler)

for i in range(0, N_TRIAL, N_BATCH):
    trials = []
    for j in range(N_BATCH):
        trials.append(study.ask())
    print("before suggest_float")
    X = [trial.suggest_float("x", -10, 10) for trial in trials]
    print("doing suggest_float")

    states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
    trials_temp = study._get_trials(deepcopy=False, states=states)

    for t in trials_temp:
        # print("trial_id", t._trial_id)
        # print("trial.params", t.params)
        # print("sampler._get_params", sampler._get_params(t))
        # print()
        pass

    print("Y before suggest_float")
    Y = [trial.suggest_float("y", -10, 10) for trial in trials]
    print("after suggest_float")
    for j in range(N_BATCH):
        study.tell(trials[j], X[j] ** 2 + Y[j] ** 2)

    # Skip first random sampling.
    # if i > 0:
    #     plt.plot(X, Y, ".")
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.savefig(f"{multivariate}-{constant_liar}.png")
# plt.clf()