import optuna
from optuna.importance import PedAnovaImportanceEvaluator


def objective(trial):
    x1 = trial.suggest_float("x1", -10, 10)
    x2 = trial.suggest_float("x2", -10, 10)
    return x1 + x2 / 1000


study = optuna.create_study()
study.optimize(objective, n_trials=20)
# evaluator = PedAnovaImportanceEvaluator(baseline_quantile=0.9999999)
evaluator = PedAnovaImportanceEvaluator(target_quantile=0.9999999, region_quantile=1.0)
importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
print(importance)