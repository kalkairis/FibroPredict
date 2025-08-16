"""Data loaders for cohort construction."""

"""Data loaders for cohort construction."""

import datetime
import logging
import os

import numpy as np

from FibroPredict.cutils_placeholders import (
    DiagnosisLoader,
    LabTestsLoader,
    reverse_days_to_datetime,
)
from FibroPredict.config import (
    LIVER_CIRRHOSIS_DIAGNOSIS_STRINGS,
    EXCLUSION_DIAGNOSES_ICD9,
    BLOOD_TEST_FEATURES,
    BLOOD_TEST_THRESHOLDS,
)

logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
log_ = logging.getLogger(__name__)
log_.setLevel(logging.INFO)


class LiverCirrhosisDiagnosisLoader(DiagnosisLoader):
    """Loader for liver cirrhosis ICD9 diagnoses."""

    def find_icd9_codes_matches_to_terms(self):
        ret = {}
        for diagnosis in LIVER_CIRRHOSIS_DIAGNOSIS_STRINGS:
            ret[diagnosis] = [v for v in self.icd9_dicts['ICD_to_desc'].values() if diagnosis in str(v)]
        return ret

    @DiagnosisLoader.attribute_property_wrapper
    def liver_cirrhosis_icd9_dicts(self):
        ret = {k: v for k, v in self.icd9_dicts['ICD_to_desc'].items() if
               any(map(lambda diag: diag == str(v), LIVER_CIRRHOSIS_DIAGNOSIS_STRINGS))}
        return ret

    @DiagnosisLoader.attribute_property_wrapper
    def liver_cirrhosis_icd9_codes(self):
        return list(self.liver_cirrhosis_icd9_dicts.keys())

    @DiagnosisLoader.attribute_property_wrapper
    def liver_cirrhosis_icd9_cat_codes(self):
        ret = [self.icd9_dicts['ICD_to_cat'][k] for k in self.liver_cirrhosis_icd9_codes]
        return ret

    @DiagnosisLoader.data_property_wrapper
    def liver_cirrhosis_diagnoses(self):
        ret = self.all_diagnoses
        ret = ret[ret['diag_cat_code'].isin(self.liver_cirrhosis_icd9_cat_codes)]
        ret['date_start'] = ret.date_start.apply(reverse_days_to_datetime,
                                                 meta=('date_start', 'datetime64[ns]'))
        ret['date_end'] = ret.date_end.apply(reverse_days_to_datetime,
                                             meta=('date_end', 'datetime64[ns]'))
        ret = ret.compute()
        log_.info(
            f"Found a total of {len(ret)} liver cirrhosis diagnosis for {len(ret.pid.unique())} individuals"
        )
        return ret

    @DiagnosisLoader.attribute_property_wrapper
    def liver_cirrhosis_exclusion_icd9_dicts(self):
        ret = {k: self.icd9_dicts['ICD_to_desc'][k] for k in EXCLUSION_DIAGNOSES_ICD9}
        ret.update(self.liver_cirrhosis_icd9_dicts)
        return ret

    @DiagnosisLoader.attribute_property_wrapper
    def liver_cirrhosis_exclusion_icd9_cat_codes(self):
        ret = [self.icd9_dicts['ICD_to_cat'][k] for k in self.liver_cirrhosis_exclusion_icd9_dicts.keys()]
        return ret

    @DiagnosisLoader.data_property_wrapper
    def liver_cirrhosis_exclusion_dates(self):
        ret = self.all_diagnoses
        ret = ret[ret['diag_cat_code'].isin(self.liver_cirrhosis_exclusion_icd9_cat_codes)]
        ret['date_start'] = ret.date_start.apply(reverse_days_to_datetime,
                                                 meta=('date_start', 'datetime64[ns]'))
        ret = ret.groupby('pid')['date_start'].min().reset_index().compute()
        log_.info(
            f"Found a total of {len(ret)} liver cirrhosis diagnosis for {len(ret.pid.unique())} individuals"
        )
        return ret


class LiverCirrhosisLabTests(LabTestsLoader):
    """Loader for liver cirrhosis laboratory tests."""

    def __init__(self, working_dir, run_date, overwrite=False, **kwargs):
        if isinstance(run_date, datetime.datetime):
            self.run_time = run_date
        elif isinstance(run_date, str):
            self.run_time = datetime.datetime.strptime(run_date, '%Y-%m-%d')
        else:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

        run_working_dir = os.path.join(working_dir, 'run_at_' + self.run_time.strftime('%Y_%m_%d'))
        super().__init__(run_working_dir, overwrite, **kwargs)

    @LabTestsLoader.attribute_property_wrapper
    def lab_tests(self):
        ret = super().lab_tests
        ret = ret[ret.test_datetime.lt(self.run_time)]
        ret = ret[ret.value.notnull()]
        return ret

    def get_limits_from_cat_code(self, cat_code):
        inverse_blood_test_features = {v: k for k, v in BLOOD_TEST_FEATURES.items()}
        return BLOOD_TEST_THRESHOLDS[
            inverse_blood_test_features[self.cat_code_to_test_long_desc[cat_code]]
        ]

    def get_last_lab_test(self, cat_codes, filter=None, **kwargs):  # noqa: A002  # unused-argument
        ret = self.lab_tests[self.lab_tests['test_code'].isin(cat_codes)]
        if len(cat_codes) > 1:
            raise Exception("Did not implement this")
        limits = {cat_code: self.get_limits_from_cat_code(cat_code=cat_code) for cat_code in cat_codes}
        for cat_code, cat_code_limits in limits.items():
            ret = ret[ret[ret.test_code.eq(cat_code)].value.between(
                cat_code_limits.get('min', -np.inf), cat_code_limits.get('max', np.inf)
            )]

        ret = ret.compute().sort_values(by='test_datetime').groupby('pid').last().reset_index()
        return ret

    @LabTestsLoader.attribute_property_wrapper
    def blood_tests_cat_codes(self):
        blood_tests = BLOOD_TEST_FEATURES
        blood_test_names_to_cat_codes = {
            k: self.test_long_desc_to_cat_code[v] for k, v in blood_tests.items()
        }
        return blood_test_names_to_cat_codes

