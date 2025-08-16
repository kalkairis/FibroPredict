"""Command-line interface for constructing the survival analysis cohort.

Creates the cohort for a single index date and writes the results to disk by
delegating to the ``construct_single_index_date_cohort`` function.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from FibroPredict.analysis.survival_analysis.ConstructCohort.cohort_construction_single_index_date import (
    construct_single_index_date_cohort,
)
from FibroPredict.analysis.survival_analysis.config import MODEL_START_DATE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_date = (
        MODEL_START_DATE.to_pydatetime()
        if hasattr(MODEL_START_DATE, "to_pydatetime")
        else MODEL_START_DATE
    )
    parser = argparse.ArgumentParser(
        description="Create survival analysis cohort for a single index date."
    )
    parser.add_argument(
        "--index-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=default_date,
        help="Index date in YYYY-MM-DD format (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Construct the cohort for the requested index date."""
    args = parse_args()
    construct_single_index_date_cohort(
        index_date=args.index_date, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
