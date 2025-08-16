import pandas as pd

from FibroPredict.cutils_placeholders import load_events
from FibroPredict.analysis.survival_analysis.config import (
    MODEL_START_DATE,
    END_RUN_DATE,
    MODEL_UPDATE_FREQUENCY,
)
from FibroPredict.config import DATA_DIR


def index_date_generator():
    index_date = MODEL_START_DATE
    while index_date < END_RUN_DATE:
        yield index_date
        index_date += MODEL_UPDATE_FREQUENCY


def get_registration_dates(pids=[], start_date=None, end_date=None):
    """
    Look at load_events:
    - Leaving: events['event_code'] == 0
    - Joining: events['event_code'] == 1
    """
    events = load_events(load_dir=DATA_DIR, pids=pids)
    if start_date is not None:
        events = events[events.event_datetime.ge(start_date)]
    if end_date is not None:
        events = events[events.event_datetime.le(end_date)]
    last_joining_date = events[events.event_code.eq(1)].groupby('pid')['event_datetime'].max().dt.date.compute()
    last_leaving_date = events[events.event_code.eq(0)].groupby('pid')['event_datetime'].max().dt.date.compute()
    last_dates = pd.merge(last_joining_date, last_leaving_date, left_index=True, right_index=True, how='outer').fillna(
        pd.NaT)
    last_dates.columns = ['joining_date', 'leaving_date']
    return last_dates.astype('datetime64[ns]')

