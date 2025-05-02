import optuna
import numpy as np
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def partial_nan_objective(trial):
    """An objective function where some intermediate values are NaN."""
    x = trial.suggest_float('x', -10, 10)
    trial_number = trial.number
    
    # Report some intermediate values
    for step in range(10):
        value = x ** 2 + step
        
        # Make some values NaN based on certain conditions
        if step > 0 and trial_number > 1:
            logger.info(f"Trial {trial.number}, Step {step}: Reporting NaN")
            trial.report(float('nan'), step)
        else:
            logger.info(f"Trial {trial.number}, Step {step}: Reporting {value}")
            trial.report(value, step)
            
        # Check if trial should be pruned
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at step {step}")
            raise optuna.exceptions.TrialPruned()
    
    return x ** 2

def test_patient_pruner_with_nan_values():
    """Test how PatientPruner handles NaN values in different configurations."""
    
    # Create a MedianPruner to be used as base pruner for PatientPruner
    median_pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0)

    logger.info("\n=== Test 2: Study with NaN intermediate values ===")
    study = optuna.create_study(
        pruner=optuna.pruners.PatientPruner(median_pruner, patience=3),
        direction="minimize"
    )
    study.optimize(partial_nan_objective, n_trials=10)
    
    # Print results
    logger.info("Completed trials with intermediate values:")
    for trial in study.trials:
        logger.info(f"Trial {trial.number}: State={trial.state}, Value={trial.value}")
        if trial.intermediate_values:
            logger.info(f"  Intermediate values: {trial.intermediate_values}")
    
if __name__ == "__main__":
    test_patient_pruner_with_nan_values()