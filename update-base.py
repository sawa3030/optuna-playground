import optuna
from optuna.samplers import NSGAIISampler

storage_path = 'sqlite:///optuna_test_6066.db'

import warnings

# 全てのWarningを表示
warnings.simplefilter("always")

def objective(trial):
    test_param_1 = trial.suggest_categorical(
        "test_param_1",
        [False, True]
    )
    
    value = 100 * (1 if test_param_1 else 0)
       
    return value

# Do a study, this will fill the first 10 initial trial_ids in the database
study_1 = optuna.create_study(
        direction='maximize',
        load_if_exists=True,
        storage=storage_path,
        study_name='optuna_test_6066_study_1')

study_1.optimize(objective, n_trials=10, catch=(RuntimeError, ValueError, AssertionError))

# Run another study, this way the trial_id does not match the index
study_2 = optuna.create_study(
        direction='maximize',
        sampler=NSGAIISampler(
            population_size=3,
            mutation_prob=0.1,
            crossover_prob=0.9,
            swapping_prob=0.5),
        load_if_exists=True,
        storage=storage_path,
        study_name='optuna_test_6066_study_2')

study_2.optimize(objective, n_trials=10, catch=(RuntimeError, ValueError, AssertionError))
