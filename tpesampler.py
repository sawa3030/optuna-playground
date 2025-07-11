import matplotlib.pyplot as plt
import optuna

N_TRIAL = 50
N_BATCH = 10

for multivariate in (False, True):
    for constant_liar in (False, True):
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
            X = [trial.suggest_float("x", -10, 10) for trial in trials]
            Y = [trial.suggest_float("y", -10, 10) for trial in trials]
            for j in range(N_BATCH):
                study.tell(trials[j], X[j] ** 2 + Y[j] ** 2)

            # Skip first random sampling.
            if i > 0:
                plt.plot(X, Y, ".")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.savefig(f"now-{multivariate}-{constant_liar}.png")
        plt.clf()