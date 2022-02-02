# -*- coding: utf-8 -*-

"""Top-level package for RDT."""


__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.6.3.dev3'

import numpy as np
import pandas as pd

from rdt import transformers
from rdt.hyper_transformer import HyperTransformer

__all__ = [
    'HyperTransformer',
    'transformers'
]


def get_demo(num_rows=5):
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
    # Hard code first five rows
    login_dates = ['2021-06-26', '2021-02-10', 'NAT', '2020-09-26', '2020-12-22']
    last_login = [np.datetime64(i) for i in login_dates]
    email_optin = pd.Series([False, False, False, True, np.nan], dtype='boolean')
    credit_card = ['VISA', 'VISA', 'AMEX', np.nan, 'DISCOVER']
    age = [29, 18, 21, 45, 32]
    dollars_spent = [99.99, np.nan, 2.50, 25.00, 19.99]

    data = pd.DataFrame({
        'last_login': last_login,
        'email_optin': email_optin,
        'credit_card': credit_card,
        'age': age,
        'dollars_spent': dollars_spent
    })

    if num_rows <= 5:
        return data.iloc[:num_rows]

    # Randomly generate the remaining rows
    np.random.seed(42)
    num_rows -= 5

    login_dates = np.array([
        np.datetime64('2000-01-01') + np.timedelta64(np.random.randint(0, 10000), 'D')
        for _ in range(num_rows)
    ])
    login_dates[np.random.random(size=num_rows) > 0.8] = np.datetime64('NaT')

    email_optin = np.random.choice([True, False, np.nan], size=num_rows)
    email_optin = pd.Series(email_optin, dtype='boolean')

    credit_card = np.random.choice(['VISA', 'AMEX', np.nan, 'DISCOVER'], size=num_rows)
    age = np.random.randint(18, 100, size=num_rows)

    dollars_spent = np.around(np.random.uniform(0, 100, size=num_rows), decimals=2)
    dollars_spent[np.random.random(size=num_rows) > 0.8] = np.nan

    return data.append(pd.DataFrame({
        'last_login': login_dates,
        'email_optin': email_optin,
        'credit_card': credit_card,
        'age': age,
        'dollars_spent': dollars_spent
    }), ignore_index=True)
