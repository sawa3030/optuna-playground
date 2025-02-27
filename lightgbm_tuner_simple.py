"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM tuner.

In this example, we optimize the validation log loss of cancer detection.

"""

import numpy as np
import optuna.integration.lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# if __name__ == "__main__":
#     data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)
    # dtrain = lgb.Dataset(train_x, label=train_y)
    # dval = lgb.Dataset(val_x, label=val_y)

#     params = {
#         "objective": "binary",
#         "metric": "binary_logloss",
#         "verbosity": -1,
#         "boosting_type": "gbdt",
#     }

#     model = lgb.train(
#         params,
#         dtrain,
#         valid_sets=[dtrain, dval],
#         callbacks=[early_stopping(100), log_evaluation(100)],
#     )

#     prediction = np.rint(model.predict(val_x, num_iteration=model.best_iteration))
#     accuracy = accuracy_score(val_y, prediction)

#     best_params = model.params
#     print("Best params:", best_params)
#     print("  Accuracy = {}".format(accuracy))
#     print("  Params: ")
#     for key, value in best_params.items():
#         print("    {}: {}".format(key, value))

ts = time.time()

dtrain = lgb.Dataset(x_train, label=y_train)
eval_data = lgb.Dataset(x_val, label=y_val)

param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
    }

best = lgb.train(param, 
                 dtrain,
                 valid_sets=eval_data,
                 early_stopping_rounds=100)

time.time() - ts

# time: 2945.9576

"""
###実際にチューニングしてくれているパラメータ###

param = {
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
"""
