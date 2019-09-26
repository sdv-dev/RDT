import numpy as np
import pandas as pd
from faker import Faker

from rdt.transformers.base import BaseTransformer

ANONYMIZE_MAP = {}

class CategoricalTransformer(BaseTransformer):

    def __init__(self, anonymize=None):
        self.anonymize = anonymize
        self.mapping = dict()

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        uniques = data.unique()

        if self.anonymize is not None:
            fake_data = [self.anonymize() for x in range(len(uniques))]
            ANONYMIZE_MAP[id(self)] = {k: v for k, v in zip(uniques, fake_data)}
            mapping = dict(zip(fake_data, range(len(fake_data))))

        else:
            mapping = dict(zip(uniques, range(len(uniques))))

        reverse_mapping = {v: k for k, v in mapping.items()}

        self.vect_func = np.vectorize(mapping.get)
        self.vect_revert_func = np.vectorize(reverse_mapping.get)

    def transform(self, data):
        if isinstance(data, pd.Series):
            data = data.to_numpy()

        if self.anonymize is not None:
            vfunc = np.vectorize(ANONYMIZE_MAP[id(self)].get)
            data = vfunc(data)

        return self.vect_func(data)

    def reverse_transform(self, data):
        return self.vect_revert_func(data)
