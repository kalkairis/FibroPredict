import os

import pandas as pd

from FibroPredict.config import LIVER_CIRRHOSIS_WORKING_DIR

SURVIVAL_ANALYSIS_WORKING_DIR = os.path.join(LIVER_CIRRHOSIS_WORKING_DIR, 'survival_analysis')
OUTPUT_DIR = os.path.join(SURVIVAL_ANALYSIS_WORKING_DIR, 'output')

MODEL_START_DATE = globals().get('MODEL_START_DATE', pd.datetime(2005, 1, 1))
END_RUN_DATE = globals().get('END_RUN_DATE', pd.datetime(2016, 1, 1))
MODEL_UPDATE_FREQUENCY_YEARS = globals().get('MODEL_UPDATE_FREQUENCY_YEARS', 5)
MODEL_UPDATE_FREQUENCY = globals().get('MODEL_UPDATE_FREQUENCY', pd.DateOffset(years=MODEL_UPDATE_FREQUENCY_YEARS))
BASELINE_TO_INDEX_YEARS = globals().get('BASELINE_TO_INDEX_YEARS', pd.DateOffset(years=1))
FOLLOWUP_YEARS = globals().get('FOLLOWUP_YEARS', pd.DateOffset(years=5))

