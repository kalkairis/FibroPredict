"""Train models and generate predictions over multiple run dates."""

import pickle
import datetime
import logging
import os

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from FibroPredict.cutils_placeholders import BaseLoader
from FibroPredict.config import (
    MODEL_PREDICTION_TIME,
    LIVER_CIRRHOSIS_WORKING_DIR,
    MIN_RUN_DATE,
    MODEL_UPDATE_FREQUENCY,
    DATA_DIR,
)
from FibroPredict.cohort.temporal_cohort import SingleTemporalCohort

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
log_ = logging.getLogger(__name__)
log_.setLevel(logging.INFO)


class JointRunTimePredictions(BaseLoader):
    def __init__(self, working_dir, model_prediction_date, overwrite=False, **kwargs):
        try:
            self.run_time = datetime.datetime.strptime(model_prediction_date, '%Y-%m-%d')
            self.min_model_date = kwargs.get('min_model_date', MIN_RUN_DATE)
            self.min_model_date = datetime.datetime.strptime(self.min_model_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

        run_working_dir = os.path.join(working_dir, 'model_prediction_at_' + self.run_time.strftime('%Y_%m_%d'))
        self.model_update_frequency = kwargs.get('model_update_frequency', MODEL_UPDATE_FREQUENCY)

        self._overwrite = overwrite
        self._working_dir = run_working_dir
        os.makedirs(self._working_dir, exist_ok=True)
        self._visualizations_dir = kwargs.get('visualizations_dir', os.path.join(self._working_dir, 'visualizations'))
        os.makedirs(self._visualizations_dir, exist_ok=True)
        self.models_dir = os.path.join(self.working_dir, 'models_dir')
        os.makedirs(self.models_dir, exist_ok=True)
        self.kwargs = kwargs

    @BaseLoader.attribute_property_wrapper
    def run_dates(self):
        ret = [self.run_time]
        while ret[-1] - self.model_update_frequency >= self.min_model_date:
            ret.append(ret[-1] - self.model_update_frequency)
        return ret

    @BaseLoader.attribute_property_wrapper
    def temporal_cohorts(self):
        ret = {run_date: SingleTemporalCohort(self.working_dir, run_date, self.overwrite, **self.kwargs) for run_date in
               self.run_dates}
        return ret

    def get_model_df(self, is_input=False, is_outcome=False):
        assert is_input or is_outcome, IOError("Either is_input or is_output must be True")
        assert not (is_input and is_outcome), IOError("Both is_input and is_output cannot be True")
        dfs = {run_date: temporal_cohort.__getattribute__(f'single_run_{"input" if is_input else "outcome"}') for
               run_date, temporal_cohort in self.temporal_cohorts.items()}

        def add_run_date_to_df(run_date, df):
            df['run_date'] = run_date
            df = df.reset_index().set_index(['pid', 'run_date'])
            return df

        dfs = [add_run_date_to_df(run_date, dfs[run_date]) for run_date in list(sorted(self.temporal_cohorts.keys()))]
        ret_dfs = pd.concat(dfs).reset_index().set_index(['pid', 'run_date']).sort_index()
        if is_outcome:
            ret_dfs = ret_dfs['outcome']
        return ret_dfs

    @BaseLoader.data_property_wrapper
    def input_df(self):
        return self.get_model_df(is_input=True)

    @BaseLoader.data_property_wrapper
    def outcome_df(self):
        return self.get_model_df(is_outcome=True)

    def get_train_input_output(self):
        x_df = self.input_df
        y_df = self.outcome_df

        log_.info(f"Complete shapes: input:{x_df.shape}, output: {y_df.shape}")
        assert x_df.shape[0] == y_df.shape[0], Exception(
            f"Size of input and output Data Frames is not the same. {x_df.shape[0]}, {y_df.shape[0]}")

        return x_df, y_df

    @BaseLoader.data_property_wrapper
    def sample_weight(self):
        ret_index = self.input_df.reset_index()[['pid', 'run_date']]
        ret_index['run_date'] = pd.to_datetime(ret_index['run_date'])
        pid_to_weight = (self
                         .input_df
                         .reset_index()
                         .groupby('pid')
                         .run_date
                         .count()
                         .apply(lambda x: 1 / x)
                         .rename(columns={'run_date': 'sample_weight'})
                         .reset_index())
        ret = pd.merge(ret_index, pid_to_weight, on='pid', how='left').set_index(['pid', 'run_date'])
        return ret

    def load_model(self, model_run_date):
        with open(self.model_path_from_date(model_run_date), 'rb') as f:
            return pickle.load(f)

    def model_path_from_date(self, model_run_date):
        model_name = '_'.join(['model', str(model_run_date), 'dump'])
        return os.path.join(self.models_dir, model_name)

    def save_model(self, model, model_run_date):
        with open(self.model_path_from_date(model_run_date), 'wb') as f:
            pickle.dump(model, f)


    @BaseLoader.data_property_wrapper
    def predict(self, predictor_class=None, imputer_class=None, **kwargs):
        x = self.input_df.reset_index().set_index(['pid', 'run_date'])
        y = self.outcome_df.reset_index().set_index(['pid', 'run_date'])
        sample_weight = self.sample_weight
        sample_weight = sample_weight.reset_index().set_index(['pid', 'run_date'])
        results = {}

        # Turn values to numeric
        x['age_at_run'] = x['age_at_run'].apply(lambda age: pd.Timedelta(age).days)
        x['year_of_birth'] = pd.to_datetime(x.birth_datetime).apply(lambda birth: birth.year)
        x['month_of_birth'] = pd.to_datetime(x.birth_datetime).apply(lambda birth: birth.month)
        x.drop(columns='birth_datetime', inplace=True)

        # Remove test dates
        x.drop(columns=[col for col in x.columns if 'test_datetime' in col], inplace=True)

        run_dates = sorted(x.reset_index()['run_date'].unique())
        for i, run_date in enumerate(run_dates):
            if i == 0:
                continue

            # Create data
            try:
                x_train = x[x.reset_index()['run_date'].lt(run_date).values]
                y_train = y.loc[x_train.index]
                x_test = x[x.reset_index()['run_date'].eq(run_date).values]
                y_test = y.loc[x_test.index]
            except Exception as e:
                x_train = x[x.reset_index()['run_date'].lt(run_date).values]
                y_train = y.loc[x_train.index]
                x_test = x[x.reset_index()['run_date'].eq(run_date).values]
                y_test = y.loc[x_test.index]

            # Create model
            if predictor_class is None:
                # estimator = RidgeClassifier(class_weight={True: round(1 / (y.sum() / y.size)), False: 1})
                estimator = XGBClassifier(scale_pos_weight=(y_train.shape[
                                                                0] - y_train.values.sum()) / y_train.values.sum(),
                                          tree_method='exact')
            else:
                raise Exception("implement this")
            if imputer_class is None:
                imputer = SimpleImputer()
            else:
                raise Exception("implement this")
            pipe = Pipeline([('impute', imputer), ('estimate', estimator)])

            # Train
            try:
                pipe.fit(x_train, y_train.values.ravel())
            except Exception as e:
                pipe.fit(x_train, y_train.values.ravel())
            self.save_model(pipe, run_date)

            # Test
            predicted_y = pipe.predict(x_test)
            predicted_proba = pipe.predict_proba(x_test)

            # Save results
            curr_results = pd.DataFrame(index=x_test.index)
            curr_results['true_y'] = y_test
            curr_results['predicted_y'] = predicted_y
            curr_results['predict_proba_0'] = predicted_proba[:, 0]
            curr_results['predict_proba'] = predicted_proba[:, 1]
            results[run_date] = curr_results

        results = pd.concat(results.values())
        results['capped_proba'] = results.predict_proba.clip(lower=0, upper=1)
        results['sample_weight'] = sample_weight.loc[results.index]
        return results

    def create_precision_recall_graph(self):
        prediction_results = self.predict
        precision, recall, threshold = precision_recall_curve(prediction_results.true_y,
                                                              prediction_results.capped_proba,
                                                              sample_weight=prediction_results['sample_weight'].values)
        return {'precision': precision, 'recall': recall, 'threshold': threshold}


if __name__ == "__main__":
    prediction_class = JointRunTimePredictions(working_dir=LIVER_CIRRHOSIS_WORKING_DIR,
                                               model_prediction_date=MODEL_PREDICTION_TIME, load_dir=DATA_DIR)
    print(prediction_class.outcome_df.head())
    prediction_class.create_precision_recall_graph()

