import optuna
import numpy as np
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def nan_objective(trial):
    """An objective function that returns NaN for every 3rd trial."""
    trial_number = trial.number
    
    # Every 3rd trial returns NaN
    if trial_number % 3 == 0:
        logger.info(f"Trial {trial_number}: Returning NaN")
        return float('nan')
    else:
        value = trial.suggest_float('x', -10, 10)
        result = value ** 2  # Simple quadratic function
        logger.info(f"Trial {trial_number}: Returning {result}")
        return result

def partial_nan_objective(trial):
    """An objective function where some intermediate values are NaN."""
    x = trial.suggest_float('x', -10, 10)
    trial_number = trial.number
    
    # Report some intermediate values
    for step in range(10):
        value = x ** 2 + step
        
        # Make every 3rd step return NaN
        if step < 5 and trial_number > 2:
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
    
    # # Test 1: Basic study with NaN return values
    # logger.info("\n=== Test 1: Basic study with NaN final values ===")
    # study1 = optuna.create_study(
    #     pruner=optuna.pruners.PatientPruner(median_pruner, patience=2),
    #     direction="minimize"
    # )
    # study1.optimize(nan_objective, n_trials=10)
    
    # # Print results
    # logger.info("Completed trials:")
    # for trial in study1.trials:
    #     logger.info(f"Trial {trial.number}: State={trial.state}, Value={trial.value}")
    
    # Test 2: Study with NaN intermediate values
    logger.info("\n=== Test 2: Study with NaN intermediate values ===")
    study2 = optuna.create_study(
        pruner=optuna.pruners.PatientPruner(median_pruner, patience=5),
        direction="minimize"
    )
    study2.optimize(partial_nan_objective, n_trials=10)
    
    # Print results
    logger.info("Completed trials with intermediate values:")
    for trial in study2.trials:
        logger.info(f"Trial {trial.number}: State={trial.state}, Value={trial.value}")
        if trial.intermediate_values:
            logger.info(f"  Intermediate values: {trial.intermediate_values}")
    
    # # Test 3: Zero patience with NaN values
    # logger.info("\n=== Test 3: Zero patience with NaN values ===")
    # study3 = optuna.create_study(
    #     pruner=optuna.pruners.PatientPruner(median_pruner, patience=0),
    #     direction="minimize"
    # )
    # study3.optimize(partial_nan_objective, n_trials=10)
    
    # Print results
    # logger.info("Completed trials with zero patience:")
    # for trial in study3.trials:
    #     logger.info(f"Trial {trial.number}: State={trial.state}, Value={trial.value}")
    #     if trial.intermediate_values:
    #         logger.info(f"  Intermediate values: {trial.intermediate_values}")

if __name__ == "__main__":
    test_patient_pruner_with_nan_values()