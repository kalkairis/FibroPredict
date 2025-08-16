"""Build a single temporal cohort for model training."""

import datetime
import os

import pandas as pd

from FibroPredict.cutils_placeholders import DemographicsLoader
from FibroPredict.cohort.data_loaders import (
    LiverCirrhosisDiagnosisLoader,
    LiverCirrhosisLabTests,
)
from FibroPredict.config import MAX_AGE, MIN_AGE, MODEL_UPDATE_FREQUENCY


class SingleTemporalCohort(DemographicsLoader):
    def __init__(self, working_dir, run_date, overwrite=False, diagnosis_loader=None, lab_tests_loader=None, **kwargs):
        if isinstance(run_date, datetime.datetime):
            self.run_date = run_date
        elif isinstance(run_date, str):
            self.run_date = datetime.datetime.strptime(run_date, '%Y-%m-%d')
        else:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        if globals().get('DATA_FOLDER'):
            kwargs['load_dir'] = kwargs.get('load_dir', globals()['DATA_FOLDER'])
        self.diagnosis_loader = diagnosis_loader
        if self.diagnosis_loader is None:
            self.diagnosis_loader = LiverCirrhosisDiagnosisLoader(working_dir, overwrite, **kwargs)
        self.lab_tests_loader = lab_tests_loader
        if self.lab_tests_loader is None:
            self.lab_tests_loader = LiverCirrhosisLabTests(working_dir, run_date, overwrite, **kwargs)
        run_working_dir = os.path.join(working_dir, 'run_at_' + self.run_date.strftime('%Y_%m_%d'))
        super(SingleTemporalCohort, self).__init__(run_working_dir, overwrite, **kwargs)

    @DemographicsLoader.attribute_property_wrapper
    def age_filter_pids(self):
        ret = self.all_demographics
        ret['age_at_run'] = (-(ret['birth_datetime'] - self.run_date))
        ret = ret[ret['age_at_run'].ge(datetime.timedelta(days=MIN_AGE * 365.25))]
        ret = ret[ret['age_at_run'].lt(datetime.timedelta(days=MAX_AGE * 365.25))]
        # Remove dead people
        ret = ret[(ret['death_datetime'].isnull()) | (ret['death_datetime'].ge(self.run_date))]
        return ret.pid.unique().values.compute()

    @DemographicsLoader.attribute_property_wrapper
    def diagnosis_exclusion_at_run_time(self):
        diags = self.diagnosis_loader.liver_cirrhosis_exclusion_dates.astype(
            {'pid': 'int64', 'date_start': 'datetime64[ns]'})
        ret = diags[diags['date_start'].lt(self.run_date)].pid.unique()
        return ret

    @DemographicsLoader.data_property_wrapper
    def single_run_cohort(self):
        dem = self.all_demographics

        # Inclusion:
        dem = dem[dem['pid'].isin(self.age_filter_pids)]

        # Exclusion:
        dem = dem[~dem['pid'].isin(self.diagnosis_exclusion_at_run_time)]

        # Remove patients missing values in any of ['hb', 'plt', 'wbc']
        all_patient_input = self.single_run_all_patients_input  # check issue with pid=1075
        dem = dem.compute().set_index('pid')
        dem = dem.loc[(all_patient_input[[('value', 'hb'), ('value', 'plt'), ('value', 'wbc')]]
                       .notnull()
                       .all(axis=1)
                       .loc[dem.index.values]
                       .fillna(False))]
        return dem

    @DemographicsLoader.data_property_wrapper
    def single_run_all_patients_input(self):
        ret = self.lab_tests_loader.blood_tests_cat_codes
        ret = {k: self.lab_tests_loader.get_last_lab_test([v]) for k, v in ret.items()}
        for k, v in ret.items():
            v['test_name'] = k
            v.set_index('pid', inplace=True)
        ret = pd.concat(list(ret.values()), axis=0)
        ret = ret[['value', 'test_datetime', 'test_name']]
        ret = ret.reset_index().set_index(['pid', 'test_name']).unstack(level=1)
        return ret

    @DemographicsLoader.data_property_wrapper
    def single_run_input(self):
        ret = self.single_run_all_patients_input

        # add age and gender to input
        ret = pd.merge(
            self.single_run_cohort[['birth_datetime', 'age_at_run', 'is_male']],
            ret,
            how='left',
            left_index=True,
            right_index=True,
        )

        # Add difference between ALT and AST if exists
        alt_col_name = [col for col in ret.columns if ('value' in col) and ('alt' in col)][0]
        ast_col_name = [col for col in ret.columns if ('value' in col) and ('ast' in col)][0]
        ret['alt_ast_diff'] = ret[alt_col_name] - ret[ast_col_name]
        ret['alt_ast_ratio'] = ret[alt_col_name] / ret[ast_col_name]

        return ret

    @DemographicsLoader.data_property_wrapper
    def single_run_outcome(self):
        diags_in_label = self.diagnosis_loader.liver_cirrhosis_diagnoses.astype(
            {'pid': 'int64', 'date_start': 'datetime64[ns]'}
        )
        start_time = self.run_date
        end_time = self.run_date + MODEL_UPDATE_FREQUENCY
        diags_in_label = diags_in_label[diags_in_label.date_start.between(start_time, end_time)]
        diagnosed_pids = diags_in_label.pid.unique()
        ret = self.single_run_cohort
        ret['outcome'] = ret.reset_index().pid.isin(diagnosed_pids).values
        ret = ret['outcome'].to_frame().reset_index()
        return ret

