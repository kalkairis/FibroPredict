"""Plot follow-up cessation percentages by year from index date.

This script reproduces the functionality of the former
``outcomes_distributions_time_to_event.ipynb`` notebook. It calculates the
percentage of individuals who halted follow-up at each year after the index
date and saves the resulting visualization as Figure S2 in the liver cirrhosis
working directory.

The figure is stored under ``LIVER_CIRRHOSIS_WORKING_DIR`` as
``figure_S2_follow_up_halt_percent.png``.

Run this module as a script to generate the figure::

    python visualizations/outcomes_distributions_time_to_event.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from FibroPredict.config import LIVER_CIRRHOSIS_WORKING_DIR
from FibroPredict.analysis.survival_analysis.ConstructCohort.merge_cohort_constructions import (
    main as construct_cohort,
)


def _load_follow_up() -> pd.DataFrame:
    """Load the follow-up data and compute helper columns."""
    follow_up = construct_cohort()["follow_up"]
    follow_up["days_to_end_of_follow_up"] = (
        follow_up["T"] - follow_up.index.get_level_values(1)
    ).dt.days
    follow_up["years_to_end_of_follow_up"] = (
        follow_up["days_to_end_of_follow_up"].div(365.25).astype(int)
    )
    follow_up = follow_up.reset_index()
    max_follow_up = (
        follow_up.groupby("index_date")["T"]
        .max()
        .rename("max_end_of_follow_up")
        .reset_index()
    )
    follow_up = follow_up.merge(max_follow_up, on="index_date")
    return follow_up


def _aggregate_follow_up(follow_up: pd.DataFrame) -> pd.DataFrame:
    """Aggregate follow-up data to year-level percentages."""
    people_per_date = follow_up.groupby("index_date").pid.count().to_dict()

    ended_follow_up_early = follow_up[
        follow_up["T"].lt(follow_up["max_end_of_follow_up"])
    ].copy()

    ended_per_year = (
        ended_follow_up_early.groupby(["index_date", "years_to_end_of_follow_up"])
        .pid.count()
        .rename("ended")
        .to_frame()
    )
    ended_per_year["cumulative_ended"] = (
        ended_per_year.groupby(level=0)["ended"].cumsum()
    )
    ended_per_year["total_started"] = [
        people_per_date[idx_date]
        for idx_date in ended_per_year.index.get_level_values("index_date")
    ]
    ended_per_year["started_per_year"] = ended_per_year.apply(
        lambda row: row["total_started"]
        if row.name[1] == 0
        else row["total_started"] - row["cumulative_ended"] + row["ended"],
        axis=1,
    )

    per_year_counts = (
        ended_follow_up_early.groupby(
            ["index_date", "years_to_end_of_follow_up", "E"]
        )
        .pid.count()
        .rename("finished_individuals")
    )
    per_year_summary = per_year_counts.to_frame().join(
        ended_per_year["started_per_year"],
        on=["index_date", "years_to_end_of_follow_up"],
    )
    per_year_summary.rename(
        columns={"started_per_year": "started_individuals"}, inplace=True
    )
    per_year_summary.reset_index(inplace=True)
    per_year_summary["dropped_percentage"] = (
        per_year_summary["finished_individuals"].div(
            per_year_summary["started_individuals"]
        )
        * 100.0
    )
    per_year_summary["index_date"] = per_year_summary["index_date"].astype(str)
    per_year_summary["E"] = per_year_summary["E"].map(
        {0: "Censored", 1: "Diagnosed"}
    )
    return per_year_summary


def _plot_follow_up_dropoff(per_year_summary: pd.DataFrame, save_dir: str) -> str:
    """Create and save the follow-up drop-off figure."""
    g = sns.catplot(
        data=per_year_summary,
        x="years_to_end_of_follow_up",
        hue="index_date",
        y="dropped_percentage",
        col="E",
        kind="bar",
        sharey=True,
        palette="husl",
    )
    g.set_titles("{col_name}")
    g.set_axis_labels(
        x_var="Years from Index Date", y_var="Percentage of Dropped Individuals"
    )
    g.legend.set_title("Index Date")
    g.fig.subplots_adjust(bottom=0.15)
    out_path = os.path.join(save_dir, "figure_S2_follow_up_halt_percent.png")
    g.savefig(out_path, dpi=1000)
    plt.close(g.fig)
    return out_path


def main() -> None:
    follow_up = _load_follow_up()
    per_year_summary = _aggregate_follow_up(follow_up)
    os.makedirs(LIVER_CIRRHOSIS_WORKING_DIR, exist_ok=True)
    _plot_follow_up_dropoff(per_year_summary, LIVER_CIRRHOSIS_WORKING_DIR)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
