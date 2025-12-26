import matplotlib.pyplot as plt
import optuna

N_TRIAL = 50
N_BATCH = 10

for constant_liar in (None, "worst"):
# for constant_liar in (True, False):
    # sampler = optuna.samplers.TPESampler(
    #     seed=42,
    #     constant_liar=constant_liar,
    # )
    sampler = optuna.samplers.GPSampler(
        seed=42,
        constant_liar=constant_liar,
    )
    study = optuna.create_study(sampler=sampler)

    for i in range(0, N_TRIAL, N_BATCH):
        trials = []
        for j in range(N_BATCH):
            trials.append(study.ask())
        X = [trial.suggest_float("x", -10, 10) for trial in trials]
        Y = [trial.suggest_float("y", -10, 10) for trial in trials]
        for j in range(N_BATCH):
            study.tell(trials[j], X[j] ** 2 + Y[j] ** 2)

        # Skip first random sampling.
        if i > 10:
            plt.plot(X, Y, ".")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.savefig(f"tpe-constant-liar{constant_liar}.png")
    plt.clf()