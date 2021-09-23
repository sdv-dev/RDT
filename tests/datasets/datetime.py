"""Dataset Generators for datetime transformers."""

import datetime

import numpy as np

from tests.datasets.base import BaseDatasetGenerator
from tests.datasets.utils import add_nans


class DatetimeGenerator(BaseDatasetGenerator):
    """Base class for generators that generate datatime data"""

    DATA_TYPE = 'datetime'


class RandomGapDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta(days=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class RandomGapSecondsDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps of seconds between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta(seconds=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class RandomGapDatetimeNaNsGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps and NaNs"""

    @staticmethod
    def generate(num_rows):
        dates = RandomGapDatetimeGenerator.generate(num_rows)
        return add_nans(dates.astype('O'))


class EqualGapHoursDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with hour gaps between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(hours=i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class EqualGapDaysDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with 1 day gaps between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class EqualGapWeeksDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with 1 week gaps between them"""

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(weeks=i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')
