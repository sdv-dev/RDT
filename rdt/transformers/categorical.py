import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm

from rdt.transformers.base import BaseTransformer

MAPS = {}


class CategoricalTransformer(BaseTransformer):
    """Transformer for categorical data.

    This transformer expects a ``column`` ``pandas.Series`` of any dtype in
    a ``pandas.DataFrame`` table. On transform, it will map categorical values
    into the interval [0, 1], back and forth mapping all the unique values close
    to their frequency in the fit data. This means that two instances of the same
    category may not be transformed into the same number.

    On ``reverse_transform`` it will transform any value close to the frenquency
    to their related category. This behavior is to allow the transformed data to be
    modelled and the sampled data to be ``reverse_transformed``.

    Args:
        anonymize (str, tuple or list):
            Anonymization category. ``None`` disables anonymization. Defaults to ``None``.
    """

    mapping = None

    def __init__(self, anonymize=False):
        self.anonymize = anonymize

    def get_faker(self):
        """Return the faker object to anonymize data.

        Returns:
            function:
                Faker function to generate new data instances with ``self.anonymize`` arguments.

        Raises:
            ValueError:
                A ``ValueError`` is raised if the faker category we want don't exist.
        """

        if isinstance(self.anonymize, (tuple, list)):
            category, *args = self.anonymize
        else:
            category = self.anonymize
            args = tuple()

        try:
            faker_method = getattr(Faker(), category)

            def faker():
                return faker_method(*args)

            return faker
        except AttributeError:
            raise ValueError('Category "{}" couldn\'t be found on faker'.format(self.anonymize))

    def _anonymize(self, data):
        """Anonymize data and save in-memory the anonymized label encoding."""
        faker = self.get_faker()
        uniques = data.unique()
        fake_data = [faker() for x in range(len(uniques))]

        mapping = dict(zip(uniques, fake_data))
        MAPS[id(self)] = mapping

        return data.map(mapping)

    @staticmethod
    def _get_probabilities(data):
        """Compute probabilities for each categorical value.

        Args:
            data (pandas.Series):
                Data to compute the probabilities.

        Returns:
            dict:
                Probabilities for each categorical value (interval, mean and std).
        """
        frequencies = data.value_counts(dropna=False)
        start = 0
        end = 0
        elements = len(data)

        probabilities = dict()
        for value, frequency in frequencies.items():
            prob = frequency / elements
            end = start + prob
            interval = (start, end)
            mean = np.mean(interval)
            std = prob / 6
            probabilities[value] = (interval, mean, std)
            start = end

        return probabilities

    def fit(self, data):
        """Prepare the transformer before convert data.

        Create the mapping dict to save the label encoding. and anonymize data if needed.
        Finaly, compute the probabilities for each categorical value.

        Args:
            data (pandas.Series or numpy.array):
                Data to fit.
        """
        self.mapping = dict()

        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.anonymize:
            data = self._anonymize(data)

        self.probabilities = self._get_probabilities(data)

    def get_val(self, x):
        """Convert cat value into num between 0 and 1."""
        interval, mean, std = self.probabilities[x]
        return norm.rvs(mean, std)

    def transform(self, data):
        """Transform categorical data.

        If data is anonymized map the real values.

        Real values encoding is only available in-memory and can't be pickled.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if self.anonymize:
            data = data.map(MAPS[id(self)])

        return data.fillna(np.nan).apply(self.get_val)

    @staticmethod
    def _normalize(data):
        """Normalize data between the range [0, 1]."""
        data = data - data.astype(int)
        data[data < 0] += 1
        return data

    def reverse_transform(self, data):
        """Converts data back into original format.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            pandas.Series
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        data = self._normalize(data)

        result = pd.Series(index=data.index)

        for category, stats in self.probabilities.items():
            start, end = stats[0]
            result[(start < data) & (data < end)] = category

        return result
