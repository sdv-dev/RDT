
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

import rdt
from rdt import _find_addons, get_demo


@pytest.fixture()
def mock_rdt():
    rdt_module = sys.modules['rdt']
    rdt_mock = Mock()
    rdt_mock.submodule.__name__ = 'rdt.submodule'
    sys.modules['rdt'] = rdt_mock
    yield rdt_mock
    sys.modules['rdt'] = rdt_module


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


@patch.object(rdt, 'iter_entry_points')
def test__find_addons_module(entry_points_mock, mock_rdt):
    """Test loading an add-on."""
    # Setup
    add_on_mock = Mock(spec=ModuleType)
    entry_point = Mock()
    entry_point.name = 'rdt.submodule.entry_name'
    entry_point.load.return_value = add_on_mock
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    assert mock_rdt.submodule.entry_name == add_on_mock
    assert sys.modules['rdt.submodule.entry_name'] == add_on_mock


@patch.object(rdt, 'iter_entry_points')
def test__find_addons_object(entry_points_mock, mock_rdt):
    """Test loading an add-on."""
    # Setup
    entry_point = Mock()
    entry_point.name = 'rdt.submodule:entry_object.entry_method'
    entry_point.load.return_value = 'new_method'
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    assert mock_rdt.submodule.entry_object.entry_method == 'new_method'


@patch('warnings.warn')
@patch('rdt.iter_entry_points')
def test__find_addons_bad_addon(entry_points_mock, warning_mock):
    """Test failing to load an add-on generates a warning."""
    # Setup
    def entry_point_error():
        raise ValueError()

    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.module_name = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    msg = 'Failed to load "bad_entry_point" from "bad_module".'

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('rdt.iter_entry_points')
def test__find_addons_wrong_base(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_base.bad_entry_point'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'bad_base.bad_entry_point': expected base module to be 'rdt', found "
        "'bad_base'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('rdt.iter_entry_points')
def test__find_addons_missing_submodule(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'rdt.missing_submodule.new_submodule'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'rdt.missing_submodule.new_submodule': module 'rdt' has no attribute "
        "'missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('rdt.iter_entry_points')
def test__find_addons_module_and_object(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'rdt.missing_submodule:new_object'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'rdt.missing_submodule:new_object': cannot add 'new_object' to unknown "
        "submodule 'rdt.missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch.object(rdt, 'iter_entry_points')
def test__find_addons_missing_object(entry_points_mock, warning_mock, mock_rdt):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'rdt.submodule:missing_object.new_method'
    entry_points_mock.return_value = [bad_entry_point]
    msg = ("Failed to set 'rdt.submodule:missing_object.new_method': missing_object.")

    del mock_rdt.submodule.missing_object

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='rdt_modules')
    warning_mock.assert_called_once_with(msg)
