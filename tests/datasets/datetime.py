"""Dataset Generators for datetime transformers."""

import datetime

import numpy as np
import pandas as pd

from tests.datasets.base import BaseDatasetGenerator
from tests.datasets.utils import add_nans


class DatetimeGenerator(BaseDatasetGenerator):
    """Base class for generators that generate datatime data"""

    DATA_TYPE = 'datetime'

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-6,
                'memory': 200.0
            },
            'transform': {
                'time': 1e-6,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 2e-6,
                'memory': 500.0,
            }
        }


class RandomGapDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta(days=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 2e-06,
                'memory': 500.0
            },
            'transform': {
                'time': 1e-06,
                'memory': 300.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 1000.0,
            }
        }


class RandomGapSecondsDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps of seconds between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta(seconds=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 2e-06,
                'memory': 500.0
            },
            'transform': {
                'time': 1e-06,
                'memory': 300.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 1000.0,
            }
        }


class RandomGapDatetimeNaNsGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps and NaNs"""

    @staticmethod
    def generate(num_rows):
        dates = RandomGapDatetimeGenerator.generate(num_rows)
        return add_nans(dates.astype('O'))

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 2e-06,
                'memory': 500.0
            },
            'transform': {
                'time': 4e-06,
                'memory': 1000.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 1000.0,
            }
        }


class EqualGapHoursDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with hour gaps between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(hours=i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 2e-06,
                'memory': 500.0
            },
            'transform': {
                'time': 1e-06,
                'memory': 300.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 1000.0,
            }
        }


class EqualGapDaysDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with 1 day gaps between them"""

    @staticmethod
    def generate(num_rows):
        delta = datetime.timedelta

        today = min(datetime.datetime.today(), pd.Timestamp.max - delta(num_rows))
        dates = [delta(i) + today for i in range(num_rows)]

        return np.array(dates, dtype='datetime64')

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 2e-06,
                'memory': 500.0
            },
            'transform': {
                'time': 1e-06,
                'memory': 300.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 1000.0,
            }
        }


class EqualGapWeeksDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with 1 week gaps between them"""

    @staticmethod
    def generate(num_rows):
        delta = datetime.timedelta

        today = datetime.datetime.today()
        dates = [min(delta(weeks=i) + today, pd.Timestamp.max) for i in range(num_rows)]

        return np.array(dates, dtype='datetime64')

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 2e-06,
                'memory': 500.0
            },
            'transform': {
                'time': 1e-06,
                'memory': 300.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 1000.0,
            }
        }
