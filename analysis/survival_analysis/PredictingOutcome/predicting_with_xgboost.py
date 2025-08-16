import os
import pandas as pd
import xgboost
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, auc

from FibroPredict.analysis.survival_analysis.PredictingOutcome.helpers import (
    get_xy,
    train_test_split,
)
from FibroPredict.config import SURVIVAL_ANALYSIS_WORKING_DIR


def get_survival_split_input():
    """Prepare training and test splits for survival prediction."""
    df, outcome = get_xy(dropna=False)
    outcome *= df["E"].astype(bool).apply(lambda e: 1 if e else -1)
    df.drop(columns="E", inplace=True)
    return train_test_split(df, outcome)


def train_model(x, y):
    """Train an XGBoost Cox model on the provided data."""
    params = {
        "seed": 0,
        "nthread": 40,
        "objective": "survival:cox",
        "n_estimators": 100,
        "base_score": 1,
    }
    model = xgboost.XGBRegressor(**params)
    model.fit(x, y)
    return model


def evaluate(model, x, y):
    """Return c-index, AUC and the results dataframe for the given split."""
    results = pd.concat([x, y.to_frame(name="T_days")], axis=1)
    results["pred_T"] = model.predict(x)
    results["E"] = results["T_days"].ge(0).astype(int)
    results["T_days_abs"] = results["T_days"].abs()

    c_ind = concordance_index(results["T_days_abs"], -results["pred_T"], results["E"])
    fpr, tpr, _ = roc_curve(results["E"], results["pred_T"])
    auc_score = auc(fpr, tpr)
    return c_ind, auc_score, results


def main():
    pred_dir = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = get_survival_split_input()
    model = train_model(x_train, y_train)

    train_cind, train_auc, train_results = evaluate(model, x_train, y_train)
    test_cind, test_auc, test_results = evaluate(model, x_test, y_test)

    train_results.to_csv(os.path.join(pred_dir, "train_results.csv"))
    test_results.to_csv(os.path.join(pred_dir, "test_results.csv"))
    model.save_model(os.path.join(pred_dir, "model"))

    print(f"Train: {len(train_results):,}, Test: {len(test_results):,}")
    print(f"Train AUC: {train_auc:.4f}, Train c-index: {train_cind:.4f}")
    print(f"Test AUC: {test_auc:.4f}, Test c-index: {test_cind:.4f}")


if __name__ == "__main__":
    main()
