import numpy as np
import os
import pandas as pd
from FibroPredict.config import *
from FibroPredict.analysis.survival_analysis.IndexDates.helpers import (
    compute_exclusion_and_diagnosis_dates,
    get_death_dates,
    get_registration_dates,
    get_expanded_exclusions_and_diagnoses_dates,
)
from FibroPredict.cutils_placeholders import (
    get_ICD_dicts,
    load_demographics,
    load_diagnoses,
    load_events,
    load_labtests,
)
from datetime import datetime, date, timedelta

def get_follow_up_dates(date_run, overwrite=False):
    date_dir = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, date_run.strftime('%Y_%m_%d'))
    os.makedirs(date_dir, exist_ok=True)
    out_path = os.path.join(date_dir, 'follow_up_dates.csv')
    if overwrite or not os.path.exists(out_path):

        if os.path.exists(os.path.join(date_dir, 'labtests.csv')):
            pids = pd.read_csv(os.path.join(date_dir, 'labtests.csv'), index_col=0).index.values
        else:
            pids = []
        end_of_followup = date_run + FOLLOWUP_YEARS

        exclusion_dates = get_expanded_exclusions_and_diagnoses_dates()
        exclusion_dates = pd.concat([pd.to_datetime(exclusion_dates[col][
            pd.to_datetime(exclusion_dates[col]).between(
                date_run, end_of_followup, inclusive='left')]).dt.date for col in exclusion_dates.columns], 
                                    axis=1).fillna(pd.NaT).rename(
            columns={col: '_'.join(col.split('_')[1:]) for col in exclusion_dates.columns})


        death_dates = get_death_dates(pids=pids, start_date=date_run, end_date=end_of_followup)

        registration_dates = get_registration_dates(pids=pids, start_date=date_run, end_date=end_of_followup)['leaving_date'].dropna().to_frame()

        ret = pd.concat([exclusion_dates, death_dates, registration_dates], axis=1, sort=False).fillna(pd.NaT)

        if len(pids) > 0:
            ret = ret.loc[ret.index.isin(pids)].copy()

        ret['end_of_follow_up_date'] = pd.to_datetime(ret.stack()).dt.date.reset_index().sort_values(by=0).groupby('pid').first()[0]

        ret['censored_by_diagnosis'] = (ret['end_of_follow_up_date'] == ret['diagnosis_date'])
        
        ret['index_date'] = date_run
        
        ret.to_csv(out_path)
        
    ret = pd.read_csv(out_path, index_col=0).fillna(pd.NaT)
    return ret

if __name__=="__main__":
    overwrite = True
    all_exclusions = {}
    date_run = datetime.strptime(MODEL_PREDICTION_TIME, '%Y-%m-%d')
    while date_run >= datetime.strptime(MIN_RUN_DATE, '%Y-%m-%d'):
        print(f"Running on date: {date_run}")
        exclude = get_follow_up_dates(date_run, overwrite=overwrite)
        all_exclusions[date_run] = exclude
        date_run -= MODEL_UPDATE_FREQUENCY
    all_exclusions = pd.concat(list(all_exclusions.values()), axis=0)
    all_exclusions.to_csv(
        os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'follow_up_dates.csv')
    )
