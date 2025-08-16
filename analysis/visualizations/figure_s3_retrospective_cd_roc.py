"""Generate Figure S3: Retrospective cohort C/D ROC and survival curves.

This script reproduces the four panels described in the manuscript:

(A) ROC curves for Fibro-Predict (orange) and FIB-4 (blue) using all
    observations with valid FIB-4 values.
(B) ROC curves where invalid FIB-4 values are replaced with the minimum
    valid FIB-4 value.
(C) ROC curves for the subset of observations with valid FIB-4 values for
    both Fibro-Predict and FIB-4.
(D) Kaplan–Meier curves comparing subjects with and without valid FIB-4
    scores, including the log–rank test statistic.

Each panel is saved as a separate PNG file in the predictions directory.

The script assumes that prediction results are available in the directory
``SURVIVAL_ANALYSIS_WORKING_DIR/predictions`` as `train_results.csv` and
`test_results.csv` files. These files are not included in the repository but
are required to run the script.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc

from FibroPredict.config import SURVIVAL_ANALYSIS_WORKING_DIR

# Colors used across all plots
COLORS = {"Fibro-Predict": "#B15822", "FIB-4": "#284995"}


def _prediction_dir() -> str:
    """Return the directory containing prediction CSV files."""
    return os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, "predictions")


def load_predictions() -> pd.DataFrame:
    """Load train and test prediction results.

    Returns
    -------
    pd.DataFrame
        Combined dataframe containing predictions from train and test sets
        with an additional ``group`` column identifying the source.
    """

    pred_dir = _prediction_dir()
    train = pd.read_csv(
        os.path.join(pred_dir, "train_results.csv"),
        parse_dates=["index_date"],
    ).set_index(["pid", "index_date"])
    test = pd.read_csv(
        os.path.join(pred_dir, "test_results.csv"),
        parse_dates=["index_date"],
    ).set_index(["pid", "index_date"])

    train["group"] = "train"
    test["group"] = "test"
    return pd.concat([train, test])


def compute_fib4(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the FIB-4 risk score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``age``, ``ast``, ``alt`` and ``plt`` columns.

    Returns
    -------
    pd.DataFrame
        Original dataframe with an additional ``fib4`` column and a
        ``valid_fib4`` boolean column indicating availability of the
        required laboratory values.
    """

    fib4 = df["age"] * df["ast"] / (df["plt"] * np.sqrt(df["alt"]))
    df = df.copy()
    df["fib4"] = fib4
    df["valid_fib4"] = ~df["fib4"].isna()
    return df


def roc_for(df: pd.DataFrame, score_col: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC for the given score column."""
    fpr, tpr, _ = roc_curve(df["E"], df[score_col])
    return fpr, tpr, auc(fpr, tpr)


def plot_scenario(df: pd.DataFrame, scenario: str, output_name: str) -> None:
    """Plot ROC curves for a given scenario and save to file.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction results with FIB-4 computations.
    scenario : str
        One of ``"A"``, ``"B"`` or ``"C"`` as described in the module
        docstring.
    output_name : str
        File name for the output PNG.
    """

    fig, ax = plt.subplots()
    for method, score_col in [("Fibro-Predict", "pred_T"), ("FIB-4", "fib4")]:
        for group, lstyle in [("train", "--"), ("test", "-")]:
            gdf = df[df["group"] == group]

            if scenario == "A":
                if method == "FIB-4":
                    gdf = gdf[gdf["valid_fib4"]]
            elif scenario == "B":
                if method == "FIB-4":
                    fib_min = df.loc[df["valid_fib4"], "fib4"].min()
                    gdf = gdf.assign(fib4=gdf["fib4"].fillna(fib_min))
            elif scenario == "C":
                gdf = gdf[gdf["valid_fib4"]]
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            if gdf.empty:
                continue

            fpr, tpr, auc_val = roc_for(gdf, score_col)
            ax.plot(
                fpr,
                tpr,
                linestyle=lstyle,
                color=COLORS[method],
                label=f"{method} {group} (AUC={auc_val:.2f})",
            )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()

    pred_dir = _prediction_dir()
    fig.savefig(
        os.path.join(pred_dir, output_name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_kaplan_meier(df: pd.DataFrame) -> None:
    """Plot Kaplan–Meier curves and log–rank test for FIB-4 availability."""
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()

    groups = {"valid": df["valid_fib4"], "invalid": ~df["valid_fib4"]}
    for name, mask in groups.items():
        kmf.fit(df.loc[mask, "T_days_abs"] / 365.25, df.loc[mask, "E"], label=name)
        kmf.plot(ax=ax)

    result = logrank_test(
        df.loc[groups["valid"], "T_days_abs"],
        df.loc[groups["invalid"], "T_days_abs"],
        event_observed_A=df.loc[groups["valid"], "E"],
        event_observed_B=df.loc[groups["invalid"], "E"],
    )
    ax.set_xlabel("Years from index")
    ax.set_ylabel("Survival probability")
    ax.set_title(
        f"log-rank $\chi^2$={result.test_statistic:.0f}, p={result.p_value:.1e}"
    )

    pred_dir = _prediction_dir()
    fig.savefig(
        os.path.join(pred_dir, "figure_s3d_km.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    df = load_predictions()
    df = compute_fib4(df)

    plot_scenario(df, "A", "figure_s3a_roc.png")
    plot_scenario(df, "B", "figure_s3b_roc.png")
    plot_scenario(df, "C", "figure_s3c_roc.png")
    plot_kaplan_meier(df)


if __name__ == "__main__":
    main()
