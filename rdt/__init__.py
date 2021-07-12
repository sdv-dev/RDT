# -*- coding: utf-8 -*-

"""Top-level package for RDT."""


__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.5.0'

import numpy as np
import pandas as pd

from rdt import transformers
from rdt.hyper_transformer import HyperTransformer

__all__ = [
    'HyperTransformer',
    'transformers'
]


def get_demo(dtypes=('int', 'float', 'str', 'datetime'), nans=0.2, size=10):
    """Generate random demo data with multiple data types.

    Args:
        dtypes (tuple or list):
            Data types to include in the generated demo data. Defaults to all.
        nans (float):
            Proportion of null values to generate. Defaults to 0.2.
        size (int):
            Number of data rows to generate.

    Returns:
        pd.DataFrame
    """
    if np.isscalar(nans):
        nans = [nans] * len(dtypes)

    columns = dict()
    for count, (dtype, nan) in enumerate(zip(dtypes, nans)):
        if dtype == 'int':
            column = np.random.randint(100, size=size)
        elif dtype == 'float':
            column = np.random.random(size) * 100
        elif dtype == 'str':
            column = np.random.choice(['a', 'b', 'c', 'd'], size=size)
        elif dtype == 'datetime':
            deltas = np.random.randint(1000000, size=10)
            datetimes = np.array([np.datetime64('2019-10-13T18:34')] * size)
            column = datetimes + deltas

        column = pd.Series(column)
        nan_index = np.random.choice(range(size), size=int(size * nan), replace=False)
        column.iloc[nan_index] = np.nan

        columns['{}_{}'.format(count, dtype)] = column

    return pd.DataFrame(columns)
