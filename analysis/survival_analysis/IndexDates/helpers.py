import pandas as pd
import os
from FibroPredict.config import *
from FibroPredict.cutils_placeholders import (
    get_ICD_dicts,
    load_demographics,
    load_diagnoses,
    load_events,
)

def get_expanded_exclusions_and_diagnoses_dates(overwrite=False):
    output_path = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'expanded_exclusion_diagnoses_df.csv')
    if overwrite or not os.path.exists(output_path):
        cat_to_ICD, ICD_to_cat, cat_to_desc, ICD_to_desc = get_ICD_dicts(load_dir=DATA_DIR)
        diagnosis_cat_codes = [c for c, v in cat_to_desc.items() if str(v).upper() in LIVER_CIRRHOSIS_DIAGNOSIS_STRINGS]
        exclusion_cat_codes = [ICD_to_cat[k] for k in EXCLUSION_DIAGNOSES_ICD9]
        exclusion_cat_codes = [cat for cat in exclusion_cat_codes if cat not in diagnosis_cat_codes]

        diagnoses = load_diagnoses(load_dir=DATA_DIR)
        diagnoses = diagnoses[diagnoses.diag_cat_code.isin(diagnosis_cat_codes + exclusion_cat_codes)].compute()
        diagnoses['diagnosis_date'] = diagnoses[['datetime_start', 'datetime_end', 'visit_datetime']].min(axis=1)
        diagnoses = diagnoses.groupby(['pid', 'diag_cat_code'])['diagnosis_date'].min().dt.date.unstack(level=-1).fillna(pd.NaT).astype('datetime64[ns]')
        diagnoses.columns = pd.MultiIndex.from_tuples(list(map(lambda col: (
            'diagnosis' if col in diagnosis_cat_codes else 'exclusion', 
            cat_to_ICD[col]), diagnoses.columns)), names=['is_diagnosis', 'ICD9'])
        diagnoses = diagnoses.T.sort_index().T
        diagnoses.to_csv(output_path)
    return pd.read_csv(output_path, index_col=0, header=[0, 1], low_memory=False, parse_dates=list(range(32)))

def compute_exclusion_and_diagnosis_dates(exclusion_diagnoses_path=None, overwrite=False):
    if exclusion_diagnoses_path is None:
        exclusion_diagnoses_path = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'exclusion_diagnosis_df.csv')
    if overwrite or not os.path.exists(exclusion_diagnoses_path):
        os.makedirs(os.path.dirname(exclusion_diagnoses_path), exist_ok=True)
        cat_to_ICD, ICD_to_cat, cat_to_desc, ICD_to_desc = get_ICD_dicts(load_dir=DATA_DIR)
        exclusion_cat_codes = [
        ICD_to_cat[k] for k in EXCLUSION_DIAGNOSES_ICD9]
        diagnosis_cat_codes = [c for c, v in cat_to_desc.items() if str(v).upper() in LIVER_CIRRHOSIS_DIAGNOSIS_STRINGS]

        exclusions = load_diagnoses(load_dir=DATA_DIR)
        exclusions = exclusions[exclusions.diag_cat_code.isin(exclusion_cat_codes)].compute()
        exclusions = exclusions.groupby('pid')[['datetime_start', 'datetime_end', 'visit_datetime']].min()
        exclusions = exclusions.min(axis=1).dt.date.to_frame().rename(columns={0:'first_exclusion_date'})

        diagnoses = load_diagnoses(load_dir=DATA_DIR)
        diagnoses = diagnoses[diagnoses.diag_cat_code.isin(diagnosis_cat_codes)].compute()
        diagnoses = diagnoses.groupby('pid')[['datetime_start', 'datetime_end', 'visit_datetime']].min()
        diagnoses = diagnoses.min(axis=1).dt.date.to_frame().rename(columns={0:'first_diagnosis_date'})

        exclusion_diagnoses_df = pd.concat([exclusions, diagnoses], axis=1).fillna(pd.NaT)

        exclusion_diagnoses_df.to_csv(exclusion_diagnoses_path)
    return pd.read_csv(exclusion_diagnoses_path, index_col=0, parse_dates=True).fillna(pd.NaT)

def get_death_dates(pids=[], start_date=None, end_date=None):
    demographics = load_demographics(load_dir=DATA_DIR, pids=pids)
    demographics = demographics[demographics.death_datetime.notnull()]
    if start_date is None:
        if end_date is not None:
            demographics = demographics[demographics['death_datetime'].lt(end_date)]
    else:
        if end_date is None:
            demographics = demographics[demographics['death_datetime'].ge(end_date)]
        else:
            demographics = demographics[demographics['death_datetime'].between(start_date, end_date, inclusive='left')]
    demographics = demographics.compute()
    demographics = demographics.set_index('pid').death_datetime.dt.date.to_frame().rename(columns={'death_datetime': 'death_date'})
    return demographics

def get_registration_dates(pids=[], start_date=None, end_date=None):
    """
    Look at load_events:
    - Joining: events['event_code'] == 1
    - Leaving: events['event_code'] == 0
    """
    events = load_events(load_dir=DATA_DIR, pids=pids)
    if start_date is not None:
        events = events[events.event_datetime.ge(start_date)]
    if end_date is not None:
        events = events[events.event_datetime.lt(end_date)]
    last_joining_date = events[events.event_code.eq(1)].groupby('pid')['event_datetime'].max().dt.date.compute()
    last_leaving_date = events[events.event_code.eq(0)].groupby('pid')['event_datetime'].max().dt.date.compute()
    last_dates = pd.merge(last_joining_date, last_leaving_date, left_index=True, right_index=True, how='outer').fillna(pd.NaT)
    last_dates.columns = ['joining_date', 'leaving_date']
    return last_dates

