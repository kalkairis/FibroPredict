import numpy as np
import pandas as pd
from FibroPredict.config import *
from FibroPredict.analysis.survival_analysis.IndexDates.helpers import compute_exclusion_and_diagnosis_dates
from FibroPredict.cutils_placeholders import (
    get_ICD_dicts,
    get_lab_test_code_dicts,
    load_demographics,
    load_diagnoses,
    load_events,
    load_labtests,
)
from datetime import datetime, date, timedelta

def get_baseline_labtests(date_run):
    _, _, _, cat_code_to_test_long_desc, _, _ = get_lab_test_code_dicts(load_dir=DATA_DIR)
    test_desc_to_cat_code = {v: k for k,v in cat_code_to_test_long_desc.items()}
    blood_test_features = {test_desc_to_cat_code[v]: k for k,v in BLOOD_TEST_FEATURES.items()}
    labtests = load_labtests(load_dir=DATA_DIR)
    labtests = labtests[labtests.test_code.isin(blood_test_features.keys())]


    labtests = labtests[labtests.test_datetime.between(
        (date_run - BASELINE_TO_INDEX_YEARS).to_pydatetime(), date_run, inclusive='left')]
    labtests = labtests[labtests.apply(
        lambda row: BLOOD_TEST_THRESHOLDS[
            blood_test_features[row['test_code']]]['min']<=row['value']<BLOOD_TEST_THRESHOLDS[blood_test_features[
            row['test_code']]]['max'],
        axis=1,
        meta=(None, 'bool'))].compute()
    labtests = labtests.sort_values(by='test_datetime', ascending=False).groupby(['pid', 'test_code']).first()

    labtests['date_test'] = labtests['test_datetime'].dt.date

    labtests = labtests[['date_test', 'value']].unstack(-1)

    labtests = labtests.loc[labtests.loc[:, 
                                         [col in [
                                             k for k, v in blood_test_features.items() if v in VALUES_MUST_EXIST] for col in labtests.columns.get_level_values(1)]].notnull().all(axis=1)].copy()

    baseline_dates = labtests['date_test'].apply(lambda row: max(row[pd.notnull(row)]), axis=1)

    labtest_values = labtests['value'].rename(columns=blood_test_features)

    labtest_values['baseline_date'] = baseline_dates

    labtest_values['index_date'] = date_run
    
    labtest_values = labtest_values[~labtest_values.index.isin(get_exclusion_pids(date_run, pids=labtest_values.index.values))]
    
    return labtest_values

def get_registry_dates_prior_to_baseline(pids, date_run):
    """
      Look at load_events:
      - Leaving: events['event_code'] == 0
      - Joining: events['event_code'] == 1
    """
    events = load_events(load_dir=DATA_DIR, pids=pids)
    events = events[events.event_datetime.lt(date_run)]
    last_joining_date = events[events.event_code.eq(1)].groupby('pid')['event_datetime'].max().dt.date.compute()
    last_leaving_date = events[events.event_code.eq(0)].groupby('pid')['event_datetime'].max().dt.date.compute()
    last_dates = pd.merge(last_joining_date, last_leaving_date, left_index=True, right_index=True, how='outer')
    last_dates.columns = ['joining_date', 'leaving_date']
    last_dates['exclude'] = last_dates.apply(
        lambda row: True if pd.isnull(row['joining_date']) else (
            False if pd.isnull(row['leaving_date']) else (row['leaving_date'] > row['joining_date'])), axis=1)
    return last_dates

def get_exclusion_by_date_pids(pids, date_run):
    demographics = load_demographics(load_dir=DATA_DIR, pids=pids)
    demographics = demographics[demographics.death_datetime.notnull()]
    demographics = demographics[demographics['death_datetime'].lt(date_run)]
    return demographics.pid.unique().compute().values

def get_exclusion_by_criteria_pids(date_run):
    exclusion_diagnoses_df =  compute_exclusion_and_diagnosis_dates()
    exclude_by_criteria = pd.concat([pd.to_datetime(exclusion_diagnoses_df[col].dropna()).lt(date_run) for col in exclusion_diagnoses_df.columns],axis=1).fillna(False).any(axis=1)
    exclude_by_criteria_pids = exclude_by_criteria[exclude_by_criteria].index.values
    return exclude_by_criteria_pids

def get_exclusion_pids(date_run, pids=[]):
    exclude_by_registration = get_registry_dates_prior_to_baseline(pids, date_run)['exclude'].to_frame()
    exclude_by_death = pd.Series(index=get_exclusion_by_date_pids(pids, date_run), data=True)
    exclude_by_criteria = pd.Series(index=get_exclusion_by_criteria_pids(date_run), data=True)
    exclusion_table = pd.concat([exclude_by_registration, exclude_by_criteria, exclude_by_death], axis=1).fillna(False)
    exclusion_table.columns = ['registration', 'criteria', 'death']
    exclusion_pids = exclusion_table[exclusion_table.sum(axis=1).gt(0)].index
    return exclusion_pids

if __name__=="__main__":
    overwrite = False
    all_labtests = {}
    date_run = datetime.strptime(MODEL_PREDICTION_TIME, '%Y-%m-%d')
    while date_run >= datetime.strptime(MIN_RUN_DATE, '%Y-%m-%d'):
        print(f"Running on date: {date_run}")
        date_dir = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, date_run.strftime('%Y_%m_%d'))
        os.makedirs(date_dir, exist_ok=True)
        labtest_out_path = os.path.join(date_dir, 'labtests.csv')
        if overwrite or not os.path.exists(labtest_out_path):
            labtests = get_baseline_labtests(date_run)
            labtests.to_csv(labtest_out_path)
        else:
            labtests = pd.read_csv(labtest_out_path, index_col=0)
        all_labtests[date_run] = labtests
        date_run -= MODEL_UPDATE_FREQUENCY
    all_labtests = pd.concat(list(all_labtests.values()), axis=0)
    all_labtests.to_csv(
        os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'labtests.csv')
    )
