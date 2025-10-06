# import optuna

# ## simple optuna usage
# def objective1(trial: optuna.Trial) -> float:
#     x = trial.suggest_float("x", -5, 5)
#     y = trial.suggest_float("y", -5, 5)
#     z = trial.suggest_float("z", -5, 5)
#     return x**2 + y**2 + z

# def objective2(trial: optuna.Trial):
#     x = trial.suggest_float("x", -5, 5)
#     y = trial.suggest_float("y", -5, 5)
#     z = trial.suggest_float("z", -5, 5)
#     return x**2 + y**2 + z, x + y + z

# # storage = optuna.storages.InMemoryStorage()
# sampler = optuna.samplers.RandomSampler(seed=0)
# # sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, constant_liar=True)
# study = optuna.create_study(sampler=sampler, storage="sqlite:///tmp3.db")
# # study.optimize(objective1, n_trials=5)
# for i in range(5):
#     trial = study.ask()
#     value = objective1(trial)
#     study.tell(trial, value)
# # print(study._storage.get_all_trials(1))
# print("Best trial:", study.best_trial)

# # study.optimize(objective2, n_trials=2)
# # print("Best trial:", study.best_trial)


import optuna

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_int("y", -100, 100)
    return x**2 + y**2

sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, constant_liar=True)
study = optuna.create_study(sampler=sampler, storage="sqlite:///tmp.db")
for i in range(5):
    trial = study.ask()
    value = objective(trial)
    study.tell(trial, value)
print("Best trial:", study.best_trial)
