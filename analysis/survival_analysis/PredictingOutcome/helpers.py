import pandas as pd

from FibroPredict.analysis.survival_analysis.ConstructCohort.merge_cohort_constructions import (
    main,
)


def get_xy(dropna=True):
    merged_df = get_merged_survival_df(dropna=dropna)
    x = merged_df.drop(columns='T_days')
    y = merged_df['T_days']
    return x, y


def get_merged_survival_df(dropna=True):
    follow_up, input_df = get_unmerged_survival_dfs()
    merged_df = pd.concat([follow_up, input_df], axis=1).drop(columns='T')
    if dropna:
        merged_df.dropna(inplace=True)
    return merged_df


def get_unmerged_survival_dfs():
    ret = main()
    follow_up = ret['follow_up']
    input_df = ret['input']
    follow_up['T_days'] = follow_up.reset_index()[['index_date', 'T']].diff(axis=1)['T'].dt.days.values
    return follow_up, input_df


def get_time_event_dfs():
    merged_df = get_merged_survival_df()
    T = merged_df['T_days']
    E = merged_df['E']
    return T, E


def train_test_split(x, y):
    is_last_index = x.index.get_level_values(-1) == (x.index.get_level_values(-1).unique().max())
    x_train = x[~is_last_index].copy()
    x_test = x[is_last_index].copy()
    y_train = y.loc[x_train.index].copy()
    y_test = y.loc[x_test.index].copy()
    return x_train, y_train, x_test, y_test

