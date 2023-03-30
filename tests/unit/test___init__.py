from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt import _add_version, get_demo


def test_get_demo():
    demo = get_demo()

    assert list(demo.columns) == [
        'last_login', 'email_optin', 'credit_card', 'age', 'dollars_spent'
    ]
    assert len(demo) == 5
    assert list(demo.isna().sum(axis=0)) == [1, 1, 1, 0, 1]


def test_get_demo_many_rows():
    demo = get_demo(10)

    login_dates = pd.Series([
        '2021-06-26', '2021-02-10', 'NaT', '2020-09-26', '2020-12-22', '2019-11-27',
        '2002-05-10', '2014-10-04', '2014-03-19', '2015-09-13'
    ], dtype='datetime64[ns]')
    email_optin = [False, False, False, True, np.nan, np.nan, False, True, False, False]
    credit_card = [
        'VISA', 'VISA', 'AMEX', np.nan, 'DISCOVER', 'AMEX', 'AMEX', 'DISCOVER', 'DISCOVER', 'VISA'
    ]
    age = [29, 18, 21, 45, 32, 50, 93, 75, 39, 66]
    dollars_spent = [99.99, np.nan, 2.50, 25.00, 19.99, 52.48, 39.99, 4.67, np.nan, 23.28]

    expected = pd.DataFrame({
        'last_login': login_dates,
        'email_optin': email_optin,
        'credit_card': credit_card,
        'age': age,
        'dollars_spent': dollars_spent
    })

    pd.testing.assert_frame_equal(demo, expected)


@patch('rdt.iter_entry_points')
def test__add_version(entry_points_mock):
    # Setup
    entry_point = Mock()
    entry_points_mock.return_value = [entry_point]

    # Run
    _add_version()

    # Assert
    entry_points_mock.assert_called_once_with(name='version', group='rdt_modules')


@patch('rdt.warnings.warn')
@patch('rdt.iter_entry_points')
def test__add_version_bad_addon(entry_points_mock, warning_mock):
    # Setup
    def entry_point_error():
        raise ValueError()

    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.module = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    msg = 'Failed to load "bad_entry_point" from "bad_module".'

    # Run
    _add_version()

    # Assert
    entry_points_mock.assert_called_once_with(name='version', group='rdt_modules')
    warning_mock.assert_called_once_with(msg)
