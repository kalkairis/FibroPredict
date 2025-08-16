"""Generate SHAP beeswarm plot for the top contributing features.

This script reproduces Figure S4 from the manuscript: "Top Contributor Shapely Values in
Retrospective Validation Set". Each point in the resulting plot represents an individual
observation. The position along the x-axis indicates the SHAP value, reflecting the impact
of that feature on the model's risk prediction. Features are ranked by their mean absolute
SHAP value (overall importance). Higher SHAP values correspond to an increased predicted
risk of liver cirrhosis. Feature values are colour-coded, with red indicating higher
feature values and blue indicating lower feature values.
"""

import os

import matplotlib.pyplot as plt
import shap
import xgboost

from FibroPredict.config import SURVIVAL_ANALYSIS_WORKING_DIR
from FibroPredict.analysis.survival_analysis.PredictingOutcome.predicting_with_xgboost import (
    get_survival_split_input,
)


def main() -> None:
    """Create and save a SHAP beeswarm plot for the XGBoost survival model."""
    # Load the retrospective validation set.
    _, _, x_test, _ = get_survival_split_input()

    # Load the pre-trained model.
    model_path = os.path.join(
        SURVIVAL_ANALYSIS_WORKING_DIR, "predictions", "model"
    )
    params = {
        "seed": 0,
        "nthread": 40,
        "objective": "survival:cox",
        "n_estimators": 100,
        "base_score": 1,
    }
    model = xgboost.XGBRegressor(**params)
    model.load_model(model_path)

    # Compute SHAP values for the test set.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    # Format feature names for readability in the plot.
    display_columns = [
        (
            "Days from blood test"
            if col == "days_from_last_labtest_to_index_date"
            else col.replace("_", " ")
        )
        for col in x_test.columns
    ]
    x_display = x_test.copy()
    x_display.columns = display_columns

    # Create the beeswarm plot.
    shap.summary_plot(shap_values, x_display, show=False)
    plt.title(
        "Figure S4: Top Contributor Shapely Values in Retrospective Validation Set"
    )

    # Save figure alongside other prediction artefacts.
    out_path = os.path.join(
        SURVIVAL_ANALYSIS_WORKING_DIR, "predictions", "figure_s4_shap_beeswarm.png"
    )
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved SHAP beeswarm plot to {out_path}")


if __name__ == "__main__":
    main()
