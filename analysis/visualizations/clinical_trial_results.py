"""Generate Prospective Cohort statistics and figures.

This script loads the clinical trial results and produces statistics
and visualizations for the prospective cohort. The following
figures are generated and saved under ``LIVER_CIRRHOSIS_WORKING_DIR``:

* **Figure S5** – Liver Stiffness Distribution in Prospective Cohort (counts)
* **Figure S6** – Liver Stiffness Distribution in Prospective Cohort (percent)
* **Figure S7** – AUDIT scores

The figures are stored only in the liver cirrhosis working directory as
``figure_S5_liver_stiffness_counts.png``,
``figure_S6_liver_stiffness_percent.png`` and
``figure_S7_audit_scores.png``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from FibroPredict.config import LIVER_CIRRHOSIS_WORKING_DIR, LIVER_MAIN_DIR


def _load_invited_participants() -> pd.DataFrame:
    """Load invited participants from the clinical trial results."""
    results_path = os.path.join(
        LIVER_MAIN_DIR,
        "AfulaData",
        "clinical_trial",
        "clinical_trial_results.csv",
    )
    clinical_trial_results = pd.read_csv(results_path, index_col=0)
    invited = clinical_trial_results[
        clinical_trial_results["patient number"].notnull()
    ].copy()
    invited["Clinical Group"] = invited.is_control.astype(bool).apply(
        lambda c: "FIB-4" if c else "Model"
    )
    return clinical_trial_results, invited


def _print_statistics(clinical_trial_results: pd.DataFrame, invited: pd.DataFrame) -> None:
    """Print basic statistics for the prospective cohort."""
    print(
        f"""
Total number of individuals in clinical trial {len(clinical_trial_results)}.
Of which {clinical_trial_results.is_control.sum()} from Fib-4,
and {(~clinical_trial_results.is_control.astype(bool)).sum()} from model.

Invited {len(invited)}.
From which {invited['Clinical Group'].eq('FIB-4').sum()} from FIB-4
and {invited['Clinical Group'].eq('Model').sum()} from model.
"""
    )

    mortality_stats = (
        clinical_trial_results.groupby("is_control")["is_dead"]
        .agg(["sum", lambda d: d.mean() * 100.0])
        .round(1)
    )
    mortality_stats.rename(columns={"<lambda_0>": "percent"}, inplace=True)
    print("\nMortality statistics:\n", mortality_stats)

    disease_kpa = 12
    invited["kpa_above_disease"] = invited["kPa"].gt(disease_kpa)
    kpa_stats = invited.groupby("is_control")["kpa_above_disease"].agg(
        sum="sum", percentage=lambda d: round(d.mean() * 100.0, 2)
    )
    print("\nLiver stiffness above disease threshold:")
    print(kpa_stats)

    u, p = mannwhitneyu(
        invited[invited.is_control.astype(bool)]["kPa"],
        invited[~invited.is_control.astype(bool)]["kPa"],
    )
    print(f"\nMann-Whitney U test: statistic={u:.3f}, p-value={p:.3f}")


def _ensure_output_dir() -> str:
    os.makedirs(LIVER_CIRRHOSIS_WORKING_DIR, exist_ok=True)
    return LIVER_CIRRHOSIS_WORKING_DIR


def _figure_s5(invited: pd.DataFrame, disease_kpa: float, save_dir: str) -> None:
    bins = range(0, 75, 3)
    ax = sns.histplot(
        data=invited,
        hue="Clinical Group",
        x="kPa",
        stat="count",
        palette={"FIB-4": "#284995", "Model": "#B15822"},
        multiple="dodge",
        bins=bins,
        shrink=0.8,
        common_norm=False,
    )
    ax.axvline(disease_kpa, 0, 1, color="red")
    ax.set_xlabel("Liver Stiffness (kPa)")
    ax.set_ylabel("Count")
    out_path = os.path.join(save_dir, "figure_S5_liver_stiffness_counts.png")
    plt.savefig(out_path, dpi=1000)
    plt.close()


def _figure_s6(invited: pd.DataFrame, disease_kpa: float, save_dir: str) -> None:
    bins = range(0, 75, 3)
    g = sns.histplot(
        data=invited,
        hue="Clinical Group",
        x="kPa",
        stat="probability",
        palette={"FIB-4": "#284995", "Model": "#B15822"},
        multiple="dodge",
        bins=bins,
        shrink=0.8,
        common_norm=False,
    )
    g.set_yticklabels(["0", "10", "20", "30", "40", "50"])
    g.set_ylabel("Percent")
    g.set_xlabel("Liver Stiffness (kPa)")
    g.axvline(disease_kpa, 0, 1, color="red")
    out_path = os.path.join(save_dir, "figure_S6_liver_stiffness_percent.png")
    plt.savefig(out_path, dpi=1000)
    plt.close()


def _figure_s7(invited: pd.DataFrame, save_dir: str) -> None:
    ax = sns.boxplot(
        data=invited,
        x="Clinical Group",
        y="AUDIT score",
        palette={"FIB-4": "#284995", "Model": "#B15822"},
    )
    ax.axhline(15, 0, 1, color="red")
    out_path = os.path.join(save_dir, "figure_S7_audit_scores.png")
    plt.savefig(out_path, dpi=1000)
    plt.close()


def main() -> None:
    clinical_trial_results, invited = _load_invited_participants()
    _print_statistics(clinical_trial_results, invited)
    save_dir = _ensure_output_dir()
    disease_kpa = 12
    _figure_s5(invited, disease_kpa, save_dir)
    _figure_s6(invited, disease_kpa, save_dir)
    _figure_s7(invited, save_dir)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
