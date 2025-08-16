import os

import pandas as pd

from FibroPredict.analysis.survival_analysis.ConstructCohort.cohort_construction_single_index_date import (
    construct_single_index_date_cohort,
)
from FibroPredict.analysis.survival_analysis.ConstructCohort.first_diagnoses_dates import (
    get_first_exclusion_date,
)
from FibroPredict.analysis.survival_analysis.ConstructCohort.helpers import index_date_generator
from FibroPredict.analysis.survival_analysis.config import SURVIVAL_ANALYSIS_WORKING_DIR


def main(overwrite=False):
    # Get exclusion dates
    get_first_exclusion_date(overwrite=overwrite)
    paths = {df_name: os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, df_name + '.csv') for df_name in
             ['input', 'follow_up_extended', 'follow_up']}
    if overwrite or not all(list(map(os.path.exists, paths.values()))):
        input_df = []
        follow_up_extended = []
        follow_up = []
        # Run all mains per index date
        for index_date in index_date_generator():
            idx_input_df, idx_follow_up_extended, idx_follow_up = construct_single_index_date_cohort(index_date,
                                                                                                     overwrite)
            input_df.append(idx_input_df)
            follow_up_extended.append(idx_follow_up_extended)
            follow_up.append(idx_follow_up)
        input_df = pd.concat(input_df, axis=0)
        input_df.to_csv(paths['input'])
        follow_up_extended = pd.concat(follow_up_extended, axis=0)
        follow_up_extended.to_csv(paths['follow_up_extended'])
        follow_up = pd.concat(follow_up, axis=0)
        follow_up.to_csv(paths['follow_up'])
    ret = {'input': pd.read_csv(paths['input'], index_col=[0, 1], parse_dates=['index_date'], low_memory=False),
           'follow_up_extended': pd.read_csv(paths['follow_up_extended'], index_col=[0], low_memory=False).astype(
               'datetime64[ns]').set_index('index_date', append=True),
           'follow_up': pd.read_csv(paths['follow_up'], index_col=[0, 1], parse_dates=['index_date', 'T'],
                                    low_memory=False)}
    return ret


if __name__ == "__main__":
    main()
