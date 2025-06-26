# -*- coding: utf-8 -*-

"""Top-level package for RDT."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '1.17.1'


import sys
import warnings
from importlib.metadata import entry_points
from operator import attrgetter
from types import ModuleType

import numpy as np
import pandas as pd

from rdt import transformers
from rdt.hyper_transformer import HyperTransformer

__all__ = ['HyperTransformer', 'transformers']

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
    login_dates = pd.Series(
        ['2021-06-26', '2021-02-10', 'NAT', '2020-09-26', '2020-12-22'],
        dtype='datetime64[ns]',
    )
    email_optin = pd.Series([False, False, False, True, np.nan], dtype='object')
    credit_card = ['VISA', 'VISA', 'AMEX', np.nan, 'DISCOVER']
    age = [29, 18, 21, 45, 32]
    dollars_spent = [99.99, np.nan, 2.50, 25.00, 19.99]

    data = pd.DataFrame({
        'last_login': login_dates,
        'email_optin': email_optin,
        'credit_card': credit_card,
        'age': age,
        'dollars_spent': dollars_spent,
    })

    if num_rows <= 5:
        return data.iloc[:num_rows]

    # Randomly generate the remaining rows
    random_state = np.random.get_state()
    np.random.set_state(np.random.RandomState(RANDOM_SEED).get_state())
    try:
        num_rows -= 5

        login_dates = np.array(
            [
                np.datetime64('2000-01-01') + np.timedelta64(np.random.randint(0, 10000), 'D')
                for _ in range(num_rows)
            ],
            dtype='datetime64[ns]',
        )
        login_dates[np.random.random(size=num_rows) > 0.8] = np.datetime64('NaT')

        email_optin = pd.Series([True, False, np.nan], dtype='object').sample(
            num_rows, replace=True
        )
        credit_card = np.random.choice(['VISA', 'AMEX', np.nan, 'DISCOVER'], size=num_rows)
        age = np.random.randint(18, 100, size=num_rows)

        dollars_spent = np.around(np.random.uniform(0, 100, size=num_rows), decimals=2)
        dollars_spent[np.random.random(size=num_rows) > 0.8] = np.nan

    finally:
        np.random.set_state(random_state)

    return pd.concat(
        [
            data,
            pd.DataFrame({
                'last_login': login_dates,
                'email_optin': email_optin,
                'credit_card': credit_card,
                'age': age,
                'dollars_spent': dollars_spent,
            }),
        ],
        ignore_index=True,
    )


def _get_addon_target(addon_path_name):
    """Find the target object for the add-on.

    Args:
        addon_path_name (str):
            The add-on's name. The add-on's name should be the full path of valid Python
            identifiers (i.e. importable.module:object.attr).

    Returns:
        tuple:
            * object:
                The base module or object the add-on should be added to.
            * str:
                The name the add-on should be added to under the module or object.
    """
    module_path, _, object_path = addon_path_name.partition(':')
    module_path = module_path.split('.')

    if module_path[0] != __name__:
        msg = f"expected base module to be '{__name__}', found '{module_path[0]}'"
        raise AttributeError(msg)

    target_base = sys.modules[__name__]
    for submodule in module_path[1:-1]:
        target_base = getattr(target_base, submodule)

    addon_name = module_path[-1]
    if object_path:
        if len(module_path) > 1 and not hasattr(target_base, module_path[-1]):
            msg = f"cannot add '{object_path}' to unknown submodule '{'.'.join(module_path)}'"
            raise AttributeError(msg)

        if len(module_path) > 1:
            target_base = getattr(target_base, module_path[-1])

        split_object = object_path.split('.')
        addon_name = split_object[-1]

        if len(split_object) > 1:
            target_base = attrgetter('.'.join(split_object[:-1]))(target_base)

    return target_base, addon_name


def _find_addons():
    """Find and load all RDT add-ons.

    If the add-on is a module, we add it both to the target module and to
    ``system.modules`` so that they can be imported from the top of a file as follows:

    from top_module.addon_module import x
    """
    group = 'rdt_modules'
    try:
        eps = entry_points(group=group)  # pylint: disable=E1123
    except TypeError:
        # Load-time selection requires Python >= 3.10 or importlib_metadata >= 3.6
        eps = entry_points().get(group, [])  # pylint: disable=E1101

    for entry_point in eps:
        try:
            addon = entry_point.load()
        except Exception:  # pylint: disable=broad-exception-caught
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.value}".'
            warnings.warn(msg)
            continue

        try:
            addon_target, addon_name = _get_addon_target(entry_point.name)
        except AttributeError as error:
            msg = f"Failed to set '{entry_point.name}': {error}."
            warnings.warn(msg)
            continue

        if isinstance(addon, ModuleType):
            addon_module_name = f'{addon_target.__name__}.{addon_name}'
            if addon_module_name not in sys.modules:
                sys.modules[addon_module_name] = addon

        setattr(addon_target, addon_name, addon)


_find_addons()
