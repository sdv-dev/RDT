"""Dataset Generators for datetime transformers."""

import datetime

import numpy as np

from tests.performance.datasets.base import BaseDatasetGenerator
from tests.performance.datasets.utils import add_nans


class RandomGapDatetimeGenerator(BaseDatasetGenerator):
    """Generator that creates dates with random gaps between them"""

    TYPE = 'datetime'

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta(days=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class RandomGapSecondsDatetimeGenerator(BaseDatasetGenerator):
    """Generator that creates dates with random gaps of seconds between them"""

    TYPE = 'datetime'

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta(seconds=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class RandomGapDatetimeNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates dates with random gaps and NaNs"""

    TYPE = 'datetime'

    @staticmethod
    def generate(num_rows):
        dates = RandomGapDatetimeGenerator.generate(num_rows)
        return add_nans(dates.astype('O'))


class EqualGapHoursDatetimeGenerator(BaseDatasetGenerator):
    """Generator that creates dates with hour gaps between them"""

    TYPE = 'datetime'

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(hours=i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class EqualGapDaysDatetimeGenerator(BaseDatasetGenerator):
    """Generator that creates dates with 1 day gaps between them"""

    TYPE = 'datetime'

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')


class EqualGapWeeksDatetimeGenerator(BaseDatasetGenerator):
    """Generator that creates dates with 1 week gaps between them"""

    TYPE = 'datetime'

    @staticmethod
    def generate(num_rows):
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(weeks=i) + today for i in range(num_rows)]
        return np.array(dates, dtype='datetime64')
