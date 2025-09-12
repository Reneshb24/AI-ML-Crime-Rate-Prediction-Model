import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


train = pd.read_csv("new_train.csv")
test  = pd.read_csv("new_test_for_participants.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

X = train.drop(columns=["ID", "ViolentCrimesPerPop"])
y = train["ViolentCrimesPerPop"]
X_test = test.drop(columns=["ID"])

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

lgb_params = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.01,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 1.0,
    "random_state": 42,
    "tree_method": "hist"
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_imp, y), 1):
    print(f"\n--- Fold {fold} ---")
    X_tr, X_val = X_imp.iloc[train_idx], X_imp.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # LightGBM
    ltrain = lgb.Dataset(X_tr, label=y_tr)
    lval   = lgb.Dataset(X_val, label=y_val)

    model_lgb = lgb.train(
        params=lgb_params,
        train_set=ltrain,
        valid_sets=[lval],
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(200)]
    )

    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=5000)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,
        verbose=200
    )

    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=3,
        loss_function="RMSE",
        random_seed=42,
        verbose=500
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = (
        0.5 * model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
        + 0.3 * model_xgb.predict(X_val, iteration_range=(0, model_xgb.best_iteration))
        + 0.2 * model_cat.predict(X_val)
    )
    oof_preds[val_idx] = val_pred

    test_pred = (
        0.5 * model_lgb.predict(X_test_imp, num_iteration=model_lgb.best_iteration)
        + 0.3 * model_xgb.predict(X_test_imp, iteration_range=(0, model_xgb.best_iteration))
        + 0.2 * model_cat.predict(X_test_imp)
    )
    test_preds += test_pred / NFOLD

submission = pd.DataFrame({
    "ID": test["ID"],
    "ViolentCrimesPerPop": test_preds
})

submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")
