import os

import pandas as pd

from FibroPredict.cutils_placeholders import (
    get_lab_test_code_dicts,
    load_demographics,
    load_labtests,
)
from FibroPredict.analysis.survival_analysis.ConstructCohort.first_diagnoses_dates import (
    get_first_exclusion_date,
    get_first_liver_cirrhosis_date,
)
from FibroPredict.analysis.survival_analysis.ConstructCohort.helpers import (
    get_registration_dates,
)
from FibroPredict.analysis.survival_analysis.config import (
    SURVIVAL_ANALYSIS_WORKING_DIR,
    BASELINE_TO_INDEX_YEARS,
    FOLLOWUP_YEARS,
)
from FibroPredict.config import (
    DATA_DIR,
    MIN_AGE,
    MAX_AGE,
    BLOOD_TEST_FEATURES,
    BLOOD_TEST_THRESHOLDS,
    VALUES_MUST_EXIST,
)


def get_most_recent_lab_tests(index_date, output_dir, pids=[], overwrite=False):
    output_path = os.path.join(output_dir, 'labtests.csv')
    if overwrite or not os.path.exists(output_path):
        labtests = load_labtests(load_dir=DATA_DIR, pids=pids)

        _, _, _, cat_code_to_test_long_desc, _, _ = get_lab_test_code_dicts(load_dir=DATA_DIR)
        test_desc_to_cat_code = {v: k for k, v in cat_code_to_test_long_desc.items()}
        blood_test_features = {test_desc_to_cat_code[v]: k for k, v in BLOOD_TEST_FEATURES.items()}

        labtests = labtests[labtests['test_code'].isin(blood_test_features.keys())]
        labtests = labtests[labtests.test_datetime.between(
            (index_date - BASELINE_TO_INDEX_YEARS).to_pydatetime(), index_date, inclusive='left')]
        labtests = labtests[labtests.apply(
            lambda row: BLOOD_TEST_THRESHOLDS[
                            blood_test_features[row['test_code']]]['min'] <= row['value'] <
                        BLOOD_TEST_THRESHOLDS[blood_test_features[
                            row['test_code']]]['max'],
            axis=1,
            meta=(None, 'bool'))].compute()
        labtests = labtests.sort_values(by='test_datetime', ascending=False).groupby(['pid', 'test_code']).first()

        labtests['date_test'] = labtests['test_datetime'].dt.date

        labtests = labtests[['date_test', 'value']].unstack(-1)
        baseline_dates = labtests['date_test'].apply(lambda row: max(row[pd.notnull(row)]), axis=1)
        labtests = labtests['value'].rename(columns=blood_test_features)
        labtests['baseline_date'] = baseline_dates

        labtests['index_date'] = index_date

        labtests['days_from_last_labtest_to_index_date'] = (
                pd.to_datetime(labtests['index_date']).dt.date - pd.to_datetime(
            labtests['baseline_date']).dt.date).dt.days

        labtests.drop(columns='baseline_date', inplace=True)
        labtests.set_index('index_date', append=True, inplace=True)
        labtests.to_csv(output_path)
    return pd.read_csv(output_path, index_col=[0, 1], parse_dates=['index_date'], low_memory=False)


def get_age_inclusion_by_date(index_date, output_dir, overwrite=False):
    out_path = os.path.join(output_dir, 'demographics.csv')
    if overwrite or not os.path.exists(out_path):
        dem = load_demographics(load_dir=DATA_DIR).set_index('pid')[['birth_datetime', 'death_datetime', 'is_male']]
        dem['index_date'] = index_date
        dem = dem[~dem['death_datetime'].le(index_date)]
        dem['age'] = dem['birth_datetime'].sub(index_date).dt.days.div(365.25).mul(-1).astype(int)
        dem = dem[dem['age'].between(MIN_AGE, MAX_AGE)]
        dem = dem.compute().set_index('index_date', append=True)
        dem['date_birth'] = pd.to_datetime(dem['birth_datetime']).dt.date
        dem['date_death'] = pd.to_datetime(dem['death_datetime']).dt.date
        dem.drop(columns=['birth_datetime', 'death_datetime'], inplace=True)
        dem.to_csv(out_path)
    return pd.read_csv(out_path, index_col=[0, 1], parse_dates=['date_birth', 'date_death', 'index_date'],
                       low_memory=False)


def get_lab_tests_with_must_have_values(index_date, dir_path, lab_tests=None, pids=[], overwrite=False):
    output_path = os.path.join(dir_path, 'labtest_with_must_have_values.csv')
    if overwrite or not os.path.exists(output_path):
        if lab_tests is None:
            lab_tests = get_most_recent_lab_tests(index_date, dir_path, pids=pids)
        lab_tests = lab_tests[lab_tests[VALUES_MUST_EXIST].notnull().all(axis=1)].copy()
        lab_tests.to_csv(output_path)
    return pd.read_csv(output_path, index_col=[0, 1], parse_dates=['index_date'], low_memory=False)


def get_pids_passing_exclusion_by_diagnosis(index_date, dir_path, pids, overwrite=False):
    output_path = os.path.join(dir_path, 'exclusion_by_diagnosis.csv')
    if overwrite or not os.path.exists(output_path):
        first_exclusions = get_first_exclusion_date().drop(columns='exclusion_date')
        # Exclusion is prior to index_date
        first_exclusions = first_exclusions[first_exclusions.min(axis=1).le(index_date)]
        # Exclusion is of individuals within our cohort
        first_exclusions = first_exclusions[first_exclusions.index.isin(pids)]
        first_exclusions.to_csv(output_path)
    ret = pd.read_csv(output_path, index_col=0, low_memory=False).index.values
    ret = list(set(pids) - set(ret))
    return ret


def exclude_individuals_who_left_clalit(index_date, dir_path, pids, overwrite=False):
    output_path = os.path.join(dir_path, 'clalit_registration.csv')
    if overwrite or not os.path.exists(output_path):
        registration_dates = get_registration_dates(pids=pids, start_date=None, end_date=index_date)
        registration_dates = registration_dates[registration_dates.index.isin(pids)]
        registration_dates = registration_dates[
            ~registration_dates['leaving_date'].ge(registration_dates['joining_date'])].copy()
        registration_dates.to_csv(output_path)
    return pd.read_csv(output_path, index_col=0, low_memory=False).index.values


def get_input_df(index_date, dir_path, pids, lab_tests=None, demographics=None, overwrite=False):
    output_path = os.path.join(dir_path, 'input_table.csv')
    if overwrite or not os.path.exists(output_path):
        if lab_tests is None:
            lab_tests = get_lab_tests_with_must_have_values(index_date, dir_path)
        if demographics is None:
            demographics = get_age_inclusion_by_date(index_date, dir_path)
        demographics = demographics[['is_male', 'age']][demographics.index.get_level_values(0).isin(pids)]
        lab_tests = lab_tests[lab_tests.index.get_level_values(0).isin(pids)]
        res = pd.merge(demographics, lab_tests, left_index=True, right_index=True)
        res.to_csv(output_path)
    return pd.read_csv(output_path, index_col=[0, 1], parse_dates=['index_date'], low_memory=False)


def construct_input_df(dir_path, index_date, overwrite):
    # Step 1: get demographics
    demographics = get_age_inclusion_by_date(index_date, dir_path, overwrite=overwrite)
    pids = demographics.reset_index()['pid'].unique()
    # Step 2: get lab tests
    lab_tests = get_most_recent_lab_tests(index_date, dir_path, pids=pids, overwrite=overwrite)
    new_pids = lab_tests.index.get_level_values(0)
    assert len(set(new_pids) - set(pids)) == 0
    pids = new_pids
    # Step 3: filter lab tests for those who have values for the three test
    lab_tests = get_lab_tests_with_must_have_values(index_date, dir_path, lab_tests=lab_tests, pids=pids,
                                                    overwrite=overwrite)
    new_pids = lab_tests.index.get_level_values(0)
    assert len(set(new_pids) - set(pids)) == 0
    pids = new_pids
    # Step 4: Exclude by exclusion diagnoses prior to index date
    new_pids = get_pids_passing_exclusion_by_diagnosis(index_date, dir_path, pids, overwrite=overwrite)
    assert len(set(new_pids) - set(pids)) == 0
    pids = new_pids
    # Step 5: Exclude individuals who have left Clalit and are yet to return
    new_pids = exclude_individuals_who_left_clalit(index_date, dir_path, pids, overwrite=overwrite)
    assert len(set(new_pids) - set(pids)) == 0
    pids = new_pids
    # Step 6: merge data for a single input table
    input_df = get_input_df(index_date, dir_path, pids, lab_tests, demographics, overwrite=overwrite)
    return input_df


def construct_output_df(dir_path, index_date, pids, overwrite=False):
    output_path = os.path.join(dir_path, 'complete_follow_up_dates.csv')
    if overwrite or not os.path.exists(output_path):
        end_of_follow_up_date = index_date + FOLLOWUP_YEARS

        # Step 1: First liver cirrhosis diagnosis
        liver_cirrhosis_dates = get_first_liver_cirrhosis_date()
        liver_cirrhosis_dates = liver_cirrhosis_dates[liver_cirrhosis_dates.index.isin(pids)]
        liver_cirrhosis_dates = liver_cirrhosis_dates[
            liver_cirrhosis_dates.min(axis=1).between(index_date, end_of_follow_up_date)]
        liver_cirrhosis_dates.rename(columns={col: '_'.join(['diag', col]) for col in liver_cirrhosis_dates.columns},
                                     inplace=True)
        # Step 2: Date of death
        dem = load_demographics(pids=pids, load_dir=DATA_DIR)
        dem = dem[dem.death_datetime.between(index_date, end_of_follow_up_date)]
        dem['date_death'] = dem['death_datetime'].dt.date
        death_dates = dem.set_index('pid')['date_death'].compute().to_frame()
        # Step 3: First left censoring by diagnosis
        left_censoring_by_diag_dates = get_first_exclusion_date()
        left_censoring_by_diag_dates = left_censoring_by_diag_dates[left_censoring_by_diag_dates.index.isin(pids)]
        left_censoring_by_diag_dates.drop(
            columns=list(map(lambda col: col.split('_')[1], liver_cirrhosis_dates.columns)), inplace=True)
        left_censoring_by_diag_dates = left_censoring_by_diag_dates[
            left_censoring_by_diag_dates.min(axis=1).between(index_date, end_of_follow_up_date)]
        left_censoring_by_diag_dates.rename(
            columns={col: '_'.join(['censor', col]) for col in left_censoring_by_diag_dates.columns}, inplace=True)
        # Step 4: First leaving clalit date
        registration_dates = get_registration_dates(pids=pids,
                                                    start_date=index_date,
                                                    end_date=end_of_follow_up_date)['leaving_date'].dropna().to_frame()
        res = pd.concat([liver_cirrhosis_dates, death_dates, left_censoring_by_diag_dates, registration_dates],
                        axis=1).astype('datetime64[ns]')
        res = res.merge(pd.DataFrame(data={'pid': pids, 'end_of_follow_up': end_of_follow_up_date}).set_index('pid'),
                        left_index=True,
                        right_index=True, how='right')
        res['index_date'] = index_date
        res.set_index('index_date', append=True, inplace=True)
        res.to_csv(output_path)
    return pd.read_csv(output_path, index_col=[0], low_memory=False).astype('datetime64[ns]').set_index('index_date',
                                                                                                        append=True)
def get_follow_up_df(dir_path, output_df, overwrite=False):
    output_path = os.path.join(dir_path, 'follow_up.csv')
    if overwrite or not os.path.exists(output_path):
        follow_up_df = output_df.min(axis=1).to_frame().rename(columns={0: 'T'})

        follow_up_df['diag_date'] = output_df[[col for col in output_df if col.startswith('diag_')]].min(axis=1)

        follow_up_df['E'] = follow_up_df['T'].eq(follow_up_df['diag_date']).astype(int)

        follow_up_df.drop(columns='diag_date', inplace=True)
        follow_up_df.to_csv(output_path)
    return pd.read_csv(output_path, index_col=[0, 1], parse_dates=['index_date', 'T'], low_memory=False)

def construct_single_index_date_cohort(index_date, overwrite=False):
    dir_path = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'index_dates', index_date.strftime('%Y_%m_%d'))
    os.makedirs(dir_path, exist_ok=True)

    input_df = construct_input_df(dir_path, index_date, overwrite)
    pids = input_df.index.get_level_values(0).values

    output_df = construct_output_df(dir_path, index_date, pids, overwrite)

    follow_up = get_follow_up_df(dir_path, output_df, overwrite=overwrite)

    return input_df, output_df, follow_up

if __name__ == "__main__":
    construct_single_index_date_cohort()
