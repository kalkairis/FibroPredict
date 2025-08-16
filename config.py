import os
from datetime import date, timedelta

import yaml

import numpy as np
import pandas as pd

# //@formatter:off
if os.path.exists(__file__[:-3] + '_local.py'):
    pass
# //@formatter:on


CONFIG_PATH = os.environ.get(
    "FIBROPREDICT_CONFIG",
    os.path.join(os.path.dirname(__file__), "config.yaml"),
)
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as _cfg:
        _CONFIG = yaml.safe_load(_cfg) or {}
else:
    _CONFIG = {}

LIVER_MAIN_DIR = _CONFIG.get(
    "LIVER_MAIN_DIR", os.environ.get("LIVER_MAIN_DIR", "/path/to/liver_main_dir")
)
month_string = _CONFIG.get("MONTH_STRING", os.environ.get("MONTH_STRING", "YYYY_MM"))
LIVER_CIRRHOSIS_WORKING_DIR = os.path.join(LIVER_MAIN_DIR, month_string)
SAFE_OUTPUT_PATH = _CONFIG.get(
    "SAFE_OUTPUT_PATH", os.environ.get("SAFE_OUTPUT_PATH", "/path/to/safe_output")
)
LIVER_VISUALIZATIONS_DIR = os.path.join(LIVER_CIRRHOSIS_WORKING_DIR, "visualization")
SURVIVAL_ANALYSIS_WORKING_DIR = os.path.join(
    LIVER_CIRRHOSIS_WORKING_DIR, "survival_analysis"
)
LIVER_RAW_DATA = os.path.join(LIVER_MAIN_DIR, "AfulaData", "clinical_trial")

DATA_DIR = _CONFIG.get(
    "DATA_DIR", os.environ.get("DATA_DIR", "/path/to/preprocessed_data")
)

LIVER_CIRRHOSIS_ICD9 = globals().get('LIVER_CIRRHOSIS_ICD9', ['1550',
                                                              '452',
                                                              '4560',
                                                              '4561',
                                                              '45620',
                                                              '571',
                                                              '5712',
                                                              '5715',
                                                              '5722',
                                                              '5724',
                                                              '7895',
                                                              'D97',
                                                              'Z4291',
                                                              '78959',
                                                              '56723',
                                                              '45621',
                                                              '4562'])

LIVER_CIRRHOSIS_ICD9 = globals().get('LIVER_CIRRHOSIS_ICD9', ['1550',
                                                              '452',
                                                              '4560',
                                                              '4561',
                                                              '45620',
                                                              '571',
                                                              '5712',
                                                              '5715',
                                                              '5722',
                                                              '5724',
                                                              '7895',
                                                              'D97',
                                                              'Z4291',
                                                              '78959',
                                                              '56723',
                                                              '45621',
                                                              '4562'])

# INCLUSION EXCLUSION CRITERIA
MIN_AGE = 40
MAX_AGE = 75
EXCLUSION_DIAGNOSES_ICD9 = globals().get('EXCLUSION_DIAGNOSES_ICD9',
                                         ['V427',
                                          '07054',
                                          '07030',
                                          '1550',
                                          'V0262',
                                          'V0261',
                                          '5710',
                                          '2751',
                                          '57142',
                                          '5716',
                                          '5761',
                                          '27503',
                                          '2734',
                                          '2727',
                                          '4530',
                                          '452',
                                          '4560',
                                          '4561',
                                          '45620',
                                          '571',
                                          '5712',
                                          '5715',
                                          '5722',
                                          '5724',
                                          '7895',
                                          'D97',
                                          'Z4291',
                                          '78959',
                                          '56723',
                                          '45621',
                                          '4562',
                                          '30500', '4280', '4150', '4169', '3970', '4242'])

# TIME SPLITS
BASELINE_TO_INDEX_YEARS = globals().get('BASELINE_TO_INDEX_YEARS', pd.DateOffset(years=1))
FOLLOWUP_YEARS = globals().get('FOLLOWUP_YEARS', pd.DateOffset(years=5))
MODEL_UPDATE_FREQUENCY_YEARS = globals().get('MODEL_UPDATE_FREQUENCY_YEARS', 5)
MODEL_UPDATE_FREQUENCY = globals().get('MODEL_UPDATE_FREQUENCY', pd.DateOffset(years=MODEL_UPDATE_FREQUENCY_YEARS))
MODEL_PREDICTION_TIME = globals().get('MODEL_PREDICTION_TIME', '2015-01-01')
MIN_RUN_DATE = globals().get('MIN_RUN_DATE', '2004-01-01')
MODEL_START_DATE = globals().get('MODEL_START_DATE', pd.datetime(2005, 1, 1))
END_RUN_DATE = globals().get('END_RUN_DATE', pd.datetime(2016, 1, 1))

# Survival analysis
TEST_START_DATE = globals().get('TEST_START_DATE', '2018-01-01')

# BLOOD TESTS
BLOOD_TEST_FEATURES = globals().get('BLOOD_TEST_FEATURES', {
    # 'cbc': 'COMPLETE BLOOD COUNT',
    'hb': 'HB',  # Consider adding 'TOTAL HEMOGLOBIN'
    'plt': 'PLT',
    'wbc': 'WBC',
    'ast': 'AST (GOT)',  # consider adding 'GOT(AST)-BODY FLUID'
    'alt': 'ALT (GPT)',  # ['ALT', 'ALANINE TRANSAMINASE']
    'albumin': 'ALBUMIN',  # ['ALBUMIN', 'ALB']
    'bilirubin': 'BILIRUBIN TOTAL',  # ['BILIRUBIN-DIRECT', 'BILIRUBIN INDIRECT']
    'pt': 'PT-INR',  # ['PROTHROMBIN TIME', 'PT'] check 'PT %' or 'PT%' consider ['PT %']
    'vitamin_b12': 'VITAMIN B12',
    'glucose': 'GLUCOSE',
    'hba1c': 'HEMOGLOBIN A1C %',
    'cholesterol_total': 'CHOLESTEROL',
    'hdl': 'CHOLESTEROL- HDL',
    'ldl': 'CHOLESTEROL-LDL calc',
    'triglycerides': 'TRIGLYCERIDES',
    'total_protein': 'PROTEIN-TOTAL'})

# TODO: try adding BMI

BLOOD_TEST_THRESHOLDS = globals().get('BLOOD_TEST_THRESHOLDS', {
    'hb': {'min': 2, 'max': 20},
    'plt': {'min': 10, 'max': 1000},
    'wbc': {'min': 1, 'max': 20},
    'ast': {'min': 3, 'max': np.inf},
    'alt': {'min': 3, 'max': np.inf},
    'albumin': {'min': 1, 'max': 6},
    'bilirubin': {'min': 0.1, 'max': 100},
    'pt': {'min': 0.7, 'max': 6},
    'vitamin_b12': {'min': 5, 'max': 1500},
    'glucose': {'min': 5, 'max': 1000},
    'hba1c': {'min': 0, 'max': 20},
    'cholesterol_total': {'min': 10, 'max': np.inf},
    'hdl': {'min': 10, 'max': 100},
    'ldl': {'min': 10, 'max': np.inf},
    'triglycerides': {'min': 10, 'max': np.inf},
    'total_protein': {'min': 4, 'max': 10}})

VALUES_MUST_EXIST = ['hb', 'plt', 'wbc']

