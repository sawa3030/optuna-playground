import optuna
import optunahub

def profit_objective(trial):
    # Resource allocation problem
    workers = trial.suggest_int("workers", 1, 20)
    strategy = trial.suggest_categorical("strategy", ["aggressive", "balanced", "conservative"])

    # Calculate profit (to be maximized)
    base_profit = workers * 100
    strategy_multiplier = {"aggressive": 1.5, "balanced": 1.2, "conservative": 1.0}[strategy]
    risk_penalty = {"aggressive": 50, "balanced": 20, "conservative": 0}[strategy]

    return base_profit * strategy_multiplier - risk_penalty

# Load the hill climbing sampler
module = optunahub.load_module(
    package="samplers/hill_climbing",
    repo_owner="akchaud5",
    ref="add-hill-climbing-sampler", 
)
HillClimbingSampler = module.HillClimbingSampler
sampler = module.HillClimbingSampler(neighbor_size=6, max_restarts=8, seed=123)

# Create study for maximization
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(profit_objective, n_trials=75)

print(f"Maximum profit: {study.best_value}")
print(f"Optimal allocation: {study.best_params}")