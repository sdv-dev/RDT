# -*- coding: utf-8 -*-

"""Top-level package for RDT."""


__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '1.4.3.dev0'


import warnings
from operator import attrgetter
from sys import modules

import numpy as np
import pandas as pd
from pkg_resources import iter_entry_points

from rdt import transformers
from rdt.hyper_transformer import HyperTransformer

__all__ = [
    'HyperTransformer',
    'transformers'
]

RANDOM_SEED = 42


def get_demo(num_rows=5):
    """Generate demo data with multiple sdtypes.

    The first five rows are hard coded. The rest are randomly generated
    using ``np.random.seed(42)``.

    Args:
        num_rows (int):
            Number of data rows to generate. Defaults to 5.

    Returns:
        pd.DataFrame
    """
    # Hard code first five rows
    login_dates = pd.Series([
        '2021-06-26', '2021-02-10', 'NAT', '2020-09-26', '2020-12-22'
    ], dtype='datetime64[ns]')
    email_optin = pd.Series([False, False, False, True, np.nan], dtype='object')
    credit_card = ['VISA', 'VISA', 'AMEX', np.nan, 'DISCOVER']
    age = [29, 18, 21, 45, 32]
    dollars_spent = [99.99, np.nan, 2.50, 25.00, 19.99]

    data = pd.DataFrame({
        'last_login': login_dates,
        'email_optin': email_optin,
        'credit_card': credit_card,
        'age': age,
        'dollars_spent': dollars_spent
    })

    if num_rows <= 5:
        return data.iloc[:num_rows]

    # Randomly generate the remaining rows
    random_state = np.random.get_state()
    np.random.set_state(np.random.RandomState(RANDOM_SEED).get_state())
    try:
        num_rows -= 5

        login_dates = np.array([
            np.datetime64('2000-01-01') + np.timedelta64(np.random.randint(0, 10000), 'D')
            for _ in range(num_rows)
        ], dtype='datetime64[ns]')
        login_dates[np.random.random(size=num_rows) > 0.8] = np.datetime64('NaT')

        email_optin = pd.Series([True, False, np.nan], dtype='object').sample(
            num_rows, replace=True)
        credit_card = np.random.choice(['VISA', 'AMEX', np.nan, 'DISCOVER'], size=num_rows)
        age = np.random.randint(18, 100, size=num_rows)

        dollars_spent = np.around(np.random.uniform(0, 100, size=num_rows), decimals=2)
        dollars_spent[np.random.random(size=num_rows) > 0.8] = np.nan

    finally:
        np.random.set_state(random_state)

    return pd.concat([
        data,
        pd.DataFrame({
            'last_login': login_dates,
            'email_optin': email_optin,
            'credit_card': credit_card,
            'age': age,
            'dollars_spent': dollars_spent
        })
    ], ignore_index=True)


def _find_addons():
    """Find and load add-ons based on the given group.

    Args:
        group (str):
            The name of the entry points group to load.
        parent_globals (dict):
            The caller's global scope. Modules will be added to the parent's global scope through
            their name.
        add_all (bool):
            Whether to also add everything in the add-on's ``module.__all__`` to the parent's
            global scope. Defaults to ``False``.
    """
    group = 'rdt_modules'
    for entry_point in iter_entry_points(group=group):
        try:
            addon = entry_point.load()
        except Exception:  # pylint: disable=broad-exception-caught
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.module_name}".'
            warnings.warn(msg)
            continue

        module_path, _, object_path = entry_point.name.partition(':')
        module_path = module_path.split('.')

        if module_path[0] != __name__:
            msg = (f"Failed to load '{entry_point.name}'. Expected base module to be '{__name__}'"
                   f", found '{module_path[0]}'.")
            warnings.warn(msg)
            continue

        base_module = modules[__name__]
        for depth, submodule in enumerate(module_path[1:-1]):
            try:
                base_module = getattr(base_module, submodule)
            except AttributeError:
                msg = (f"Failed to load '{entry_point.name}'. Target submodule "
                       f"'{'.'.join(module_path[:depth + 2])}' not found.")
                warnings.warn(msg)
                continue

        if not hasattr(base_module, module_path[-1]):
            if object_path:
                msg = (f"Failed to load '{entry_point.name}'. Cannot add '{object_path}' to "
                       f"unknown submodule '{'.'.join(module_path)}'.")
                warnings.warn(msg)
                continue
            else:
                setattr(base_module, module_path[-1], addon)
        else:
            base_module = getattr(base_module, module_path[-1])

        if object_path:
            split_object = object_path.split('.')
            try:
                base_object = base_module
                if len(split_object) > 1:
                    base_object = attrgetter('.'.join(split_object[:-1]))(base_module)

                setattr(base_object, object_path[-1], addon)
            except AttributeError:
                msg = (f"Failed to load '{entry_point.name}'. Cannot find "
                       f"'{'.'.join(split_object[:-1])}' in submodule '{'.'.join(module_path)}'.")
                warnings.warn(msg)
                continue


_find_addons()
