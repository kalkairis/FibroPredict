import os

import pandas as pd

from FibroPredict.cutils_placeholders import (
    get_ICD_dicts,
    load_diagnoses,
)
from FibroPredict.analysis.survival_analysis.config import SURVIVAL_ANALYSIS_WORKING_DIR
from FibroPredict.config import (
    DATA_DIR,
    EXCLUSION_DIAGNOSES_ICD9,
    LIVER_CIRRHOSIS_ICD9,
)


def get_first_exclusion_date(overwrite=False):
    os.makedirs(SURVIVAL_ANALYSIS_WORKING_DIR, exist_ok=True)
    output_path = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'first_exclusion_date.csv')
    if overwrite or not os.path.exists(output_path):
        save_first_dates(output_path, EXCLUSION_DIAGNOSES_ICD9, 'exclusion')
    return pd.read_csv(output_path, index_col=0, parse_dates=EXCLUSION_DIAGNOSES_ICD9 + ['exclusion_date'])


def save_first_dates(output_path, icd9_codes, date_name_prefix, compute_exclusion_date=True):
    _, ICD_to_cat, _, _ = get_ICD_dicts(load_dir=DATA_DIR)
    icd9_to_cat = {icd: ICD_to_cat[icd] for icd in icd9_codes}
    diagnoses = load_diagnoses(load_dir=DATA_DIR)
    diagnoses = diagnoses[diagnoses['diag_cat_code'].isin(icd9_to_cat.values())].compute()
    diagnoses[f'{date_name_prefix}_date'] = diagnoses[['datetime_start', 'datetime_end', 'visit_datetime']].min(
        axis=1).dt.date
    diagnoses = diagnoses.groupby(['pid', 'diag_cat_code'])[f'{date_name_prefix}_date'].min().unstack(-1).astype(
        'datetime64[ns]')
    diagnoses.rename(columns={v: k for k, v in icd9_to_cat.items()}, inplace=True)
    if compute_exclusion_date:
        diagnoses['exclusion_date'] = diagnoses.min(axis=1)
    diagnoses.to_csv(output_path)


def get_first_liver_cirrhosis_date(overwrite=False):
    output_path = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'first_liver_cirrhosis_date.csv')
    if overwrite or not os.path.exists(output_path):
        save_first_dates(output_path, LIVER_CIRRHOSIS_ICD9, 'diag', compute_exclusion_date=False)
    return pd.read_csv(output_path, index_col=0, parse_dates=LIVER_CIRRHOSIS_ICD9)


if __name__ == "__main__":
    get_first_exclusion_date(overwrite=False)
    get_first_liver_cirrhosis_date(overwrite=False)
