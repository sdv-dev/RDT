import numpy as np
import pandas as pd
from faker import Faker

from rdt.transformers.base import BaseTransformer

MAPS = {}


class CategoricalTransformer(BaseTransformer):

    def __init__(self, subtype, anonymize=False):
        self.subtype = subtype
        self.anonymize = anonymize
        self.mapping = dict()

    def get_generator(self):
        """Return the generator object to anonymize data."""
        faker = Faker()

        try:
            return getattr(faker, self.subtype)
        except AttributeError:
            raise ValueError('Category "{}" couldn\'t be found on faker'.format(self.subtype))

    def _get_fake_data(self, uniques):
        """Generate fake data, map the anonymized data and return fake data."""
        faker_generator = self.get_generator()
        fake_data = [faker_generator() for x in range(len(uniques))]
        MAPS[id(self)] = {k: v for k, v in zip(uniques, fake_data)}
        return fake_data

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        uniques = data.unique()

        if self.anonymize:
            uniques = self._get_fake_data(uniques)

        mapping = dict(zip(uniques, range(len(uniques))))
        reverse_mapping = {v: k for k, v in mapping.items()}

        self.vect_func = np.vectorize(mapping.get)
        self.vect_revert_func = np.vectorize(reverse_mapping.get)

    def transform(self, data):
        if isinstance(data, pd.Series):
            data = data.to_numpy()

        if self.anonymize:
            vfunc = np.vectorize(MAPS[id(self)].get)
            data = vfunc(data)

        return pd.DataFrame({0: self.vect_func(data)})

    def reverse_transform(self, data):
        return self.vect_revert_func(data)
