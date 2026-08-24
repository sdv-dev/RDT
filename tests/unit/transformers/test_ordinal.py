import logging
import re

import numpy as np
import pandas as pd
import pytest

from rdt.errors import TransformerInputError
from rdt.transformers.ordinal import (
    OrderedLabelEncoder,
    OrderedUniformEncoder,
)

RE_SSN = re.compile(r'\d\d\d-\d\d-\d\d\d\d')


@pytest.fixture(autouse=True)
def _setup_caplog(caplog):
    """Define the logging for info message."""
    caplog.set_level(logging.INFO)


class TestOrderedUniformEncoder:
    """Unit test for the ``OrderedUniformEncoder``."""

    def test___init__(self):
        """The the ``__init__`` method.

        Passed arguments must be stored as attributes.
        """
        # Run
        transformer = OrderedUniformEncoder(order=['b', 'c', 'a', None])

        # Asserts
        pd.testing.assert_series_equal(transformer.order, pd.Series(['b', 'c', 'a', np.nan]))

    def test___init___duplicate_categories(self):
        """Test the ``__init__`` method errors if duplicate categories provided.

        Test initialization errors if duplicate categories are passed in the ``order`` parameter.
        """
        # Run / Assert
        expected_msg = (
            "The OrderedUniformEncoder has duplicate categories in the 'order' parameter. "
            'Please drop the duplicates to proceed.'
        )
        with pytest.raises(TransformerInputError, match=expected_msg):
            OrderedUniformEncoder(order=['a', 'b', 'c', 'c'])

    def test___repr___default(self):
        """Test that the ``__repr__`` method prints the custom order.

        The order should be printed as <CUSTOM> instead of the actual order.
        """
        # Setup
        transformer = OrderedUniformEncoder(order=['VISA', 'AMEX', 'DISCOVER', None])

        # Run
        stringified_transformer = transformer.__repr__()

        # Assert
        assert stringified_transformer == 'OrderedUniformEncoder(order=<CUSTOM>)'

    def test___repr___missing_value_encoding_none(self):
        """Test that the ``__repr__`` method prints non-default missing value encoding."""
        # Setup
        transformer = OrderedUniformEncoder(order=['VISA', 'AMEX'], missing_value_encoding=None)

        # Run
        stringified_transformer = transformer.__repr__()

        # Assert
        expected = 'OrderedUniformEncoder(order=<CUSTOM>, missing_value_encoding=None)'
        assert stringified_transformer == expected

    def test__get_order_alphabetical(self):
        """Test the ``_get_order`` method when ordering alphanumerically."""
        # Setup
        transformer = OrderedUniformEncoder(order=None)
        arr = np.array(['one', 'two', 'three', 'four'])

        # Run
        ordered = transformer._get_order(arr)

        # Assert
        np.testing.assert_array_equal(ordered, np.array(['four', 'one', 'three', 'two']))

    def test__get_order_alphabetical_with_nans(self):
        """Test the ``_get_order`` method with nulls when ordering alphanumerically."""
        # Setup
        transformer = OrderedUniformEncoder(order=None)
        arr = np.array(['one', 'two', 'three', np.nan, 'four'], dtype='object')

        # Run
        ordered = transformer._get_order(arr)

        # Assert
        expected = np.array(['four', 'one', 'three', 'two', np.nan], dtype='object')
        pd.testing.assert_series_equal(pd.Series(ordered), pd.Series(expected))

    def test__get_order_numerical(self):
        """Test ordering the data numerically when `order` is None"""
        # Setup
        transformer = OrderedUniformEncoder(order=None)
        arr = np.array([5, 3.11, 100, 67.8, np.nan, -2.5])

        # Run
        ordered = transformer._get_order(arr)

        # Assert
        np.testing.assert_array_equal(ordered, np.array([-2.5, 3.11, 5, 67.8, 100, np.nan]))

    def test__order_warns_mixed_dtypes(self):
        """Test transformer warns if the data contains mixed dtypes."""
        # Setup
        transformer = OrderedUniformEncoder(order=None)
        transformer.columns = ['column']
        arr = pd.Series([0, -2.0, True, 'abc'])

        # Run / Assert
        message = re.escape(
            'The data in column `column` contains mixed dtypes and no '
            '`order` is provided. Defaulting to alphabetical order.'
        )
        with pytest.warns(UserWarning, match=message):
            ordered = transformer._get_order(arr)

        np.testing.assert_array_equal(ordered, np.array([-2.0, 0, True, 'abc'], dtype='object'))

    def test__fit(self):
        """Test the ``_fit`` method."""
        # Setup
        data = pd.Series([1, 2, 3, 2, np.nan, 1, 1])
        transformer = OrderedUniformEncoder(order=[2, 3, np.nan, 1])

        # Run
        transformer._fit(data)

        # Assert
        expected_frequencies = {
            2.0: 0.2857142857142857,
            3.0: 0.14285714285714285,
            None: 0.14285714285714285,
            1.0: 0.42857142857142855,
        }
        expected_intervals = {
            2.0: [0.0, 0.2857142857142857],
            3.0: [0.2857142857142857, 0.42857142857142855],
            None: [0.42857142857142855, 0.5714285714285714],
            1.0: [0.5714285714285714, 1.0],
        }
        assert transformer.frequencies == expected_frequencies
        assert transformer.intervals == expected_intervals

    def test__fit_error(self):
        """Test the ``_fit`` method checks that data is in ``self.order``.

        If the data being fit is not in ``self.order`` an error should be raised.
        """
        # Setup
        data = pd.Series(['1', '2', '3', '2', '1', '4'], dtype='object')
        transformer = OrderedUniformEncoder(order=['2', '1'])

        # Run / Assert
        message = re.escape(
            "Unknown categories '['3', '4']'. All possible categories must be defined in the "
            "'order' parameter."
        )
        with pytest.raises(TransformerInputError, match=message):
            transformer._fit(data)

    def test__fit_info(self, caplog):
        """Test the ``_fit`` method checks that data is in ``self.order``.

        If the data being fit does not contain all the category of ``self.order``,
        an info message should be raised.
        """
        # Setup
        data = pd.DataFrame({'column_name': [1, 2, 1, 1, 2, 3, 1, 2]})
        transformer = OrderedUniformEncoder(order=[1, 2, 3, 4, 5, 6, 7])

        # Run
        transformer.fit(data, 'column_name')
        expected_message = (
            "For column 'column_name', some of the provided category "
            'values were not present in the data during fit: (4, 5, 6, +1 more).'
        )

        # Assert
        assert expected_message in caplog.text

    def test__fit_info_nan(self, caplog):
        """Test the ``_fit`` method checks that data is in ``self.order``.

        If the data being fit does not contain all the category of ``self.order``,
        an info should be raised. Check if it works for NaNs.
        """
        # Setup
        data = pd.DataFrame({'column_name': [1, 2, 1, 1, 2, 3, 1, 2]})
        transformer = OrderedUniformEncoder(order=[1, 2, 3, np.nan])

        # Run
        transformer.fit(data, 'column_name')
        expected_message = (
            "For column 'column_name', some of the provided category "
            'values were not present in the data during fit: (None).'
        )

        # Assert
        assert expected_message in caplog.text

    def test__transform(self):
        """Test the ``_transform`` method."""
        # Setup
        transformer = OrderedUniformEncoder(order=['b', 'c', 'a'])
        data = pd.Series(['a', 'b', 'b', 'a', 'a', 'c', 'a'])

        transformer.frequencies = {
            'b': 0.2857142857142858,
            'c': 0.14285714285714285,
            'a': 0.42857142857142855,
        }
        transformer.intervals = {
            'b': [0.0, 0.2857142857142857],
            'c': [0.2857142857142857, 0.42857142857142855],
            'a': [0.42857142857142855, 0.8571428571428571],
        }

        # Run
        transformed = transformer._transform(data)

        # Asserts
        for key in transformer.intervals:
            assert (transformed.loc[data == key] >= transformer.intervals[key][0]).all()
            assert (transformed.loc[data == key] < transformer.intervals[key][1]).all()

    def test__transform_error(self):
        """Test the ``_transform`` method checks that data is in ``self.order``.

        If the data being transformed is not in ``self.order`` an error should be raised.
        """
        # Setup
        data = pd.Series(['1', '2', '3', '2', '1', '4'], dtype='object')
        transformer = OrderedUniformEncoder(order=['2', '1'])

        # Run / Assert
        message = re.escape(
            "Unknown categories '['3', '4']'. All possible categories must be defined in the "
            "'order' parameter."
        )
        with pytest.raises(TransformerInputError, match=message):
            transformer._transform(data)

    def test__fit_missing_value_encoding_none_order_without_missing_values(self):
        """Test missing values are allowed outside the order when not encoded."""
        # Setup
        data = pd.Series(['a', None, 'b', np.nan])
        transformer = OrderedUniformEncoder(order=['b', 'a'], missing_value_encoding=None)

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.frequencies == {'b': 0.5, 'a': 0.5}
        assert transformer.intervals == {'b': [0.0, 0.5], 'a': [0.5, 1.0]}


class TestOrderedLabelEncoder:
    def test___init__(self):
        """The the ``__init__`` method.

        Passed arguments must be stored as attributes.
        """
        # Run
        transformer = OrderedLabelEncoder(order=['b', 'c', 'a', None], add_noise='add_noise_value')

        # Asserts
        assert transformer.add_noise == 'add_noise_value'
        pd.testing.assert_series_equal(transformer.order, pd.Series(['b', 'c', 'a', np.nan]))

    def test___init___duplicate_categories(self):
        """The the ``__init__`` method with duplicate categories in the order parameter.

        Transformer should error with ``TransformerInputError``.
        """
        # Run / Assert
        expected_msg = (
            "The OrderedLabelEncoder has duplicate categories in the 'order' parameter. "
            'Please drop the duplicates to proceed.'
        )
        with pytest.raises(TransformerInputError, match=expected_msg):
            OrderedLabelEncoder(order=['b', 'c', 'a', 'a'], add_noise='add_noise_value')

    def test__get_order_alphabetical(self):
        """Test the ``_get_order`` method sorts alphabetically when `order` is None."""
        # Setup
        transformer = OrderedLabelEncoder(order=None)
        arr = np.array(['one', 'two', 'three', 'four'])

        # Run
        ordered = transformer._get_order(arr)

        # Assert
        np.testing.assert_array_equal(ordered, np.array(['four', 'one', 'three', 'two']))

    def test__get_order_alphabetical_with_nans(self):
        """Test the ``_get_order`` method sorts alphabetically with nulls when `order` is None."""
        # Setup
        transformer = OrderedLabelEncoder(order=None)
        arr = np.array(['one', 'two', 'three', np.nan, 'four'], dtype='object')

        # Run
        ordered = transformer._get_order(arr)

        # Assert
        expected = np.array(['four', 'one', 'three', 'two', np.nan], dtype='object')
        pd.testing.assert_series_equal(pd.Series(ordered), pd.Series(expected))

    def test__get_order_numerical(self):
        """Test the ``_get_order`` method sorts numerically when ``order`` None'."""
        # Setup
        transformer = OrderedLabelEncoder(order=None)
        arr = np.array([5, 3.11, 100, 67.8, np.nan, -2.5])

        # Run
        ordered = transformer._get_order(arr)

        # Assert
        np.testing.assert_array_equal(ordered, np.array([-2.5, 3.11, 5, 67.8, 100, np.nan]))

    def test__get_order_warns_mixed_dtypes(self):
        """Test the transformer warns if the data contains mixed dtypes."""
        # Setup
        transformer = OrderedLabelEncoder(order=None)
        transformer.columns = ['column']
        arr = pd.Series([0, -2.0, True, 'abc'])

        # Run / Assert
        message = re.escape(
            'The data in column `column` contains mixed dtypes and no '
            '`order` is provided. Defaulting to alphabetical order.'
        )
        with pytest.warns(UserWarning, match=message):
            ordered = transformer._get_order(arr)

        np.testing.assert_array_equal(ordered, np.array([-2.0, 0, True, 'abc'], dtype='object'))

    def test__get_order_empty(self):
        """Test the ``_get_order`` method with empty data."""
        # Setup
        transformer = OrderedLabelEncoder(order=None)
        unique_data = np.array([])

        # Run
        ordered = transformer._get_order(unique_data)

        # Assert
        np.testing.assert_array_equal(ordered, unique_data)

    def test___repr___default(self):
        """Test that the ``__repr__`` method prints the custom order.

        The order should be printed as <CUSTOM> instead of the actual order.
        """
        # Setup
        transformer = OrderedLabelEncoder(order=['VISA', 'AMEX', 'DISCOVER', None])

        # Run
        stringified_transformer = transformer.__repr__()

        # Assert
        assert stringified_transformer == 'OrderedLabelEncoder(order=<CUSTOM>)'

    def test___repr___add_noise_true(self):
        """Test that the ``__repr__`` method prints the custom order with ``add_noise``.

        The order should be printed as <CUSTOM> instead of the actual order. If ``add_noise``
        is provided, it should be printed too.
        """
        # Setup
        transformer = OrderedLabelEncoder(order=['VISA', 'AMEX', 'DISCOVER', None], add_noise=True)

        # Run
        stringified_transformer = transformer.__repr__()

        # Assert
        assert stringified_transformer == 'OrderedLabelEncoder(order=<CUSTOM>, add_noise=True)'

    def test___repr___missing_value_encoding_none(self):
        """Test that the ``__repr__`` method prints non-default missing value encoding."""
        # Setup
        transformer = OrderedLabelEncoder(order=['VISA', 'AMEX'], missing_value_encoding=None)

        # Run
        stringified_transformer = transformer.__repr__()

        # Assert
        expected = 'OrderedLabelEncoder(order=<CUSTOM>, missing_value_encoding=None)'
        assert stringified_transformer == expected

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that a unique integer representation for each category of the data is stored
        in the ``categories_to_values`` attribute, and the reverse is stored in the
        ``values_to_categories`` attribute. The order should match the ``self.order`` indices.

        Setup:
            - create an instance of the ``OrderedLabelEncoder``.

        Input:
            - a pandas series.

        Side effects:
            - set the ``values_to_categories`` dictionary to the appropriate value.
            - set ``categories_to_values`` dictionary to the appropriate value.
        """
        # Setup
        data = pd.Series([1, 2, 3, 2, np.nan, 1])
        transformer = OrderedLabelEncoder(order=[2, 3, np.nan, 1])

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.dtype == 'float'
        expected_values_to_categories = {0: 2, 1: 3, 2: np.nan, 3: 1}
        expected_categories_to_values = {2: 0, 3: 1, 1: 3, np.nan: 2}
        for key, value in transformer.values_to_categories.items():
            assert value == expected_values_to_categories[key] or pd.isna(value)

        for key, value in transformer.categories_to_values.items():
            assert value == expected_categories_to_values.get(key) or pd.isna(key)

    def test__fit_error(self):
        """Test the ``_fit`` method checks that data is in ``self.order``.

        If the data being fit is not in ``self.order`` an error should be raised.
        """
        # Setup
        data = pd.Series(['1', '2', '3', '2', '1', '4'], dtype='object')
        transformer = OrderedLabelEncoder(order=['2', '1'])

        # Run / Assert
        message = re.escape(
            "Unknown categories '['3', '4']'. All possible categories must be defined in the "
            "'order' parameter."
        )
        with pytest.raises(TransformerInputError, match=message):
            transformer._fit(data)

    def test__fit_missing_value_encoding_none_order_without_missing_values(self):
        """Test missing values are allowed outside the order when not encoded."""
        # Setup
        data = pd.Series(['a', None, 'b', np.nan])
        transformer = OrderedLabelEncoder(order=['b', 'a'], missing_value_encoding=None)

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.values_to_categories == {0: 'b', 1: 'a'}
        assert transformer.categories_to_values == {'b': 0, 'a': 1}
