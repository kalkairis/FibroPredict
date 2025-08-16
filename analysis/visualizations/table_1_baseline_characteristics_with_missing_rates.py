"""Generate baseline characteristic tables for the manuscript.

This script reproduces the computations from the former notebook
``visualizations/table_1_baseline_characteristics_with_missing_rates.ipynb``.
It outputs two CSV files corresponding to Tables 1 and S1 in the manuscript:

1. ``baseline_characteristics_main_table_with_missing_rate.csv`` – Table 1:
   Baseline Characteristics
2. ``baseline_characteristics_supp_table_with_missing_rate.csv`` – Table S1:
   Extended baseline characteristics

The paths for these outputs are determined by the configuration in
:mod:`FibroPredict.config` and are written to
``LIVER_CIRRHOSIS_WORKING_DIR``.

Run this module as a script to generate the tables::

    python visualizations/table_1_baseline_characteristics_with_missing_rates.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from cutils.data_builder.load import load_diagnoses
from cutils.general.data_descriptions import get_ICD_dicts
from FibroPredict.analysis.survival_analysis.ConstructCohort.merge_cohort_constructions import (
    main as construct_cohort,
)
from FibroPredict.config import (
    BLOOD_TEST_FEATURES,
    DATA_DIR,
    LIVER_CIRRHOSIS_WORKING_DIR,
    LIVER_RAW_DATA,
)


def create_characteristics_table(df: pd.DataFrame, is_ehr: bool = True) -> pd.Series:
    """Create baseline characteristics table.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with baseline variables.
    is_ehr : bool, optional
        Whether the data originates from the EHR cohort, by default True.
    """

    character_dfs: list[pd.Series] = []
    subset_table = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["Group", "Characteristic"]))

    # Demographics
    subset_demographics_table: dict[tuple[str, str], str] = {}
    subset_demographics_table[("Demographics", "N")] = f"{len(df):,}"
    subset_demographics_table[("Demographics", "Age, mean (SD), y")] = (
        f"{df.age.mean():.1f} ({df.age.std():.1f})"
    )
    subset_demographics_table[("Demographics", "Sex, Female")] = (
        f"{df.is_male.eq(False).sum():,} ({df.is_male.eq(False).mul(100.0).mean():.1f})"
    )
    subset_demographics_table[("Demographics", "Sex, Male")] = (
        f"{df.is_male.sum():,} ({df.is_male.mul(100.0).mean():.1f})"
    )

    # FIB-4 score
    subset_fib4_table: dict[tuple[str, str], str] = {}
    fib4_values = df["fib_4"]
    subset_fib4_table[("FIB-4 Score", "<1.45")] = (
        f"{fib4_values.lt(1.45).sum():,} ({fib4_values.lt(1.45).mean()*100:.1f})"
    )
    subset_fib4_table[("FIB-4 Score", "1.45-3.25")] = (
        f"{fib4_values.between(1.45, 3.25).sum():,} ({fib4_values.between(1.45, 3.25).mean()*100:.1f})"
    )
    subset_fib4_table[("FIB-4 Score", ">3.25")] = (
        f"{fib4_values.gt(3.25).sum():,} ({fib4_values.gt(3.25).mean()*100:.1f})"
    )
    subset_fib4_table[("FIB-4 Score", "Missing")] = (
        f"{fib4_values.isnull().sum():,} ({fib4_values.isnull().mean()*100:.1f})"
    )

    # Laboratory tests
    subset_labtests_table: dict[tuple[str, str], str] = {}
    for feature, desc in BLOOD_TEST_FEATURES.items():
        subset_labtests_table[("Laboratory test results, mean (SD), missing %", desc)] = (
            f"{df[feature].mean():.1f} ({df[feature].std():.1f}), {df[feature].isnull().mean()*100:.0f}"
        )

    subset_table = pd.concat(
        [
            subset_table,
            pd.Series(subset_demographics_table),
            pd.Series(subset_labtests_table),
            pd.Series(subset_fib4_table),
        ]
    )
    character_dfs = [
        subset_table,
        pd.Series(subset_demographics_table),
        pd.Series(subset_labtests_table),
        pd.Series(subset_fib4_table),
    ]

    if is_ehr:
        # Prior conditions
        cat_to_ICD, ICD_to_cat, cat_to_desc, ICD_to_desc = get_ICD_dicts(load_dir=DATA_DIR)
        subset_diags_table: dict[tuple[str, str], str] = {}
        icd9_of_interest = ["401", "250"]
        single_index_diags = load_diagnoses(load_dir=DATA_DIR, pids=df.index.get_level_values("pid"))
        single_index_diags = single_index_diags[
            single_index_diags.diag_cat_code.isin(list(map(ICD_to_cat.get, icd9_of_interest)))
        ]
        single_index_diags["date_start"] = single_index_diags[
            ["datetime_start", "datetime_end", "datetime_bikur"]
        ].min(axis=1)
        single_index_diags = single_index_diags[
            single_index_diags.date_start.lt(df.index.get_level_values("index_date").min())
        ]
        single_index_diags = single_index_diags.groupby(["diag_cat_code"]).pid.nunique().compute()
        single_index_diags = single_index_diags.to_frame().reset_index()
        single_index_diags["ICD9"] = single_index_diags.diag_cat_code.apply(cat_to_ICD.get)
        single_index_diags["desc"] = single_index_diags.diag_cat_code.apply(cat_to_desc.get)
        single_index_diags["percents"] = (
            single_index_diags.pid.div(df.index.get_level_values("pid").nunique()).mul(100)
        )

        for k, v in dict(
            single_index_diags[single_index_diags.ICD9.isin(icd9_of_interest)].apply(
                lambda row: (
                    f"{row['desc']} ({row['ICD9']})",
                    f"{row['pid']:,} ({row['percents']:.1f})",
                ),
                axis=1,
            ).values
        ).items():
            subset_diags_table[("Physical health conditions", k)] = v
        character_dfs.append(pd.Series(subset_diags_table))

    subset_table = pd.concat(character_dfs)
    subset_table.rename(columns={0: "Data"})
    return subset_table


def get_ehr_baselines() -> pd.Series:
    """Baseline characteristics for the EHR cohort."""
    ehr_df = construct_cohort()["input"]
    ehr_df["fib_4"] = ehr_df.apply(
        lambda row: (row["age"] * row["ast"]) / (row["plt"] * (row["ast"] ** 0.5)), axis=1
    )
    return create_characteristics_table(ehr_df)


def get_clinical_baseline_data() -> pd.Series:
    """Baseline characteristics for the clinical cohort."""
    clinical_cohorts = pd.read_csv(
        os.path.join(LIVER_RAW_DATA, "prediction_results", "test_results.csv"), index_col=0
    )
    clinical_cohorts["fib_4"] = clinical_cohorts.apply(
        lambda row: (row["age"] * row["ast"]) / (row["plt"] * (row["ast"] ** 0.5)), axis=1
    )
    clinical_cohorts["is_male"] = clinical_cohorts["is_male"].astype(int)
    return create_characteristics_table(clinical_cohorts, is_ehr=False)


def get_ehr_baselines_by_dates() -> pd.DataFrame:
    ehr_df = construct_cohort()["input"]
    ehr_df["fib_4"] = ehr_df.apply(
        lambda row: (row["age"] * row["ast"]) / (row["plt"] * (row["ast"] ** 0.5)), axis=1
    )

    ehr_baselines_by_index_date: dict[datetime, pd.Series] = {}
    for index_date, group_df in ehr_df.reset_index(-1).groupby("index_date"):
        ehr_baselines_by_index_date[index_date] = create_characteristics_table(
            group_df.set_index("index_date", append=True)
        )
    ehr_baselines_by_index_date = pd.concat(
        [
            v.rename(columns={0: f"electronic cohort\n{k.date()}"})
            for k, v in ehr_baselines_by_index_date.items()
        ],
        axis=1,
    )
    return ehr_baselines_by_index_date


def get_clinical_cohort_by_dates() -> pd.DataFrame:
    clinical_cohorts_by_dates = {"clinical cohort stage 2": LIVER_RAW_DATA}
    clinical_cohorts_by_dates = {
        k: pd.read_csv(os.path.join(v, "prediction_results", "test_results.csv"), index_col=0)
        for k, v in clinical_cohorts_by_dates.items()
    }
    for k in clinical_cohorts_by_dates.keys():
        clinical_cohorts_by_dates[k]["cohort"] = k
    clinical_cohorts_by_dates = pd.concat(clinical_cohorts_by_dates.values(), sort=False)
    clinical_cohorts_by_dates["fib_4"] = clinical_cohorts_by_dates.apply(
        lambda row: (row["age"] * row["ast"]) / (row["plt"] * (row["ast"] ** 0.5)), axis=1
    )
    clinical_cohorts_by_dates["is_male"] = clinical_cohorts_by_dates.is_male.astype(int)
    clinical_cohorts_by_dates_baseline = []
    for cohort_name, cohort_df in clinical_cohorts_by_dates.groupby("cohort"):
        clinical_cohorts_by_dates_baseline.append(
            create_characteristics_table(cohort_df, is_ehr=False).rename(columns={0: cohort_name})
        )
    return pd.concat(clinical_cohorts_by_dates_baseline, axis=1)


def main() -> None:
    """Generate baseline characteristic tables and save them to CSV."""
    baseline_characteristics = {
        "clinical cohort": get_clinical_baseline_data(),
        "electronic cohort": get_ehr_baselines(),
    }
    ret = pd.merge(
        *[v.rename(columns={0: k}).reset_index() for k, v in baseline_characteristics.items()],
        on=["Group", "Characteristic"],
        how="outer",
    ).drop_duplicates().set_index(["Group", "Characteristic"])

    os.makedirs(LIVER_CIRRHOSIS_WORKING_DIR, exist_ok=True)
    ret.to_csv(
        os.path.join(
            LIVER_CIRRHOSIS_WORKING_DIR,
            "baseline_characteristics_main_table_with_missing_rate.csv",
        )
    )

    # Supplementary table by dates
    supp = [get_ehr_baselines_by_dates(), get_clinical_cohort_by_dates()]
    supp = pd.merge(
        *[d.reset_index() for d in supp],
        on=["Group", "Characteristic"],
        how="outer",
    ).drop_duplicates()
    supp.set_index(["Group", "Characteristic"], inplace=True)
    supp.to_csv(
        os.path.join(
            LIVER_CIRRHOSIS_WORKING_DIR,
            "baseline_characteristics_supp_table_with_missing_rate.csv",
        )
    )


if __name__ == "__main__":
    main()
