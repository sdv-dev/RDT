import logging
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt.errors import TransformerInputError
from rdt.transformers.categorical import (
    LabelEncoder,
    OneHotEncoder,
    UniformEncoder,
)

RE_SSN = re.compile(r'\d\d\d-\d\d-\d\d\d\d')


class TestUniformEncoder:
    """Test class for the UniformEncoder."""

    def test___init___bad_missing_value_encoding(self):
        """Test that the ``__init__`` raises error if ``missing_value_encoding`` is invalid."""
        # Run / Assert
        message = (
            "'missing_value_encoding' must be one of the following values: None or 'new_category'."
        )
        with pytest.raises(TransformerInputError, match=message):
            UniformEncoder(missing_value_encoding='bad_value')

    def test__fit(self):
        """Test the ``_fit`` method.

        Check the frequencies and intervals dictionnary.
        """
        # Setup
        transformer = UniformEncoder()

        # Run
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        transformer._fit(data)

        # Asserts
        expected_frequencies = {
            'foo': 0.5,
            'bar': 0.3333333333333333,
            'tar': 0.16666666666666666,
        }
        expected_intervals = {
            'foo': [0.0, 0.5],
            'bar': [0.5, 0.8333333333333333],
            'tar': [0.8333333333333333, 1.0],
        }
        assert transformer.frequencies == expected_frequencies
        assert transformer.intervals == expected_intervals

    def test_fit_with_nullable_integer_dtype(self):
        """Test that the ``fit`` method works with nullable integer columns."""
        # Setup
        data = pd.DataFrame({'example': [1, 2, 3, None]}, dtype='Int64')
        transformer = UniformEncoder()

        # Run
        transformer.fit(data=data, column='example')

        # Assert
        expected_frequencies = {
            1: 0.25,
            2: 0.25,
            3: 0.25,
            None: 0.25,
        }
        assert transformer.frequencies == expected_frequencies

    def test__fit_missing_value_encoding_none(self):
        """Test that missing values are ignored during fit when configured."""
        # Setup
        data = pd.Series(['foo', None, 'bar', np.nan])
        transformer = UniformEncoder(missing_value_encoding=None)

        # Run
        transformer._fit(data)

        # Assert
        expected_frequencies = {'foo': 0.5, 'bar': 0.5}
        expected_intervals = {'foo': [0.0, 0.5], 'bar': [0.5, 1.0]}
        assert transformer.frequencies == expected_frequencies
        assert transformer.intervals == expected_intervals

    def test__set_fitted_parameters(self):
        """Test the ``_set_fitted_parameters`` method."""
        # Setup
        transformer = UniformEncoder()
        intervals = {
            'foo': [0.0, 0.5],
            'bar': [0.5, 0.8333333333333333],
            'tar': [0.8333333333333333, 1.0],
        }

        # Run
        transformer._set_fitted_parameters('column_name', intervals, dtype='object')

        # Asserts
        assert transformer.intervals == intervals
        assert transformer.columns == ['column_name']
        assert transformer.output_columns == ['column_name']
        assert transformer.dtype == 'object'

    def test__transform(self):
        """Test the ``_transform`` method.

        Check that the labels are correctly mapped
        according to their interval.
        """
        # Setup
        transformer = UniformEncoder()
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        transformer.frequencies = {
            'foo': 0.5,
            'bar': 0.3333333333333333,
            'tar': 0.16666666666666666,
        }
        transformer.intervals = {
            'foo': [0.0, 0.5],
            'bar': [0.5, 0.8333333333333333],
            'tar': [0.8333333333333333, 1.0],
        }

        # Run
        transformed = transformer._transform(data)

        # Asserts
        for key in transformer.intervals:
            assert (transformed.loc[data == key] >= transformer.intervals[key][0]).all()
            assert (transformed.loc[data == key] < transformer.intervals[key][1]).all()

    def test__transform_missing_value_encoding_none(self):
        """Test missing values are not encoded during transform when configured."""
        # Setup
        transformer = UniformEncoder(missing_value_encoding=None)
        data = pd.Series(['foo', None, 'bar', np.nan])
        transformer.frequencies = {'foo': 0.5, 'bar': 0.5}
        transformer.intervals = {'foo': [0.0, 0.5], 'bar': [0.5, 1.0]}

        # Run
        transformed = transformer._transform(data)

        # Assert
        assert transformed.loc[[1, 3]].isna().all()
        assert (transformed.loc[data == 'foo'] >= transformer.intervals['foo'][0]).all()
        assert (transformed.loc[data == 'foo'] < transformer.intervals['foo'][1]).all()
        assert (transformed.loc[data == 'bar'] >= transformer.intervals['bar'][0]).all()
        assert (transformed.loc[data == 'bar'] < transformer.intervals['bar'][1]).all()

    def test__transform_user_warning(self):
        """Test the ``transform`` with unknown data.

        In this test ``transform`` should raise a warning due to the attempt
        of transforming data with previously unseen categories.

        Input:
        - Series with unknown categorical values
        """
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3, 4]})
        data_1 = data['col'].copy()
        data_1.loc[4] = 5
        data_2 = pd.Series([1, 2, 3, 4, 5, 'a', 7, 8, 'b'])
        transformer = UniformEncoder()
        transformer.columns = ['col']
        transformer.frequencies = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}

        transformer.intervals = {
            1: [0, 0.25],
            2: [0.25, 0.5],
            3: [0.5, 0.75],
            4: [0.75, 1],
        }

        # Run
        warning_msg_1 = re.escape(
            "The data in column 'col' contains new categories"
            " that did not appear during 'fit' (5). Assigning"
            ' them random values. If you want to model new categories,'
            " please fit the data again using 'fit'."
        )

        warning_msg_2 = re.escape(
            "The data in column 'col' contains new categories"
            " that did not appear during 'fit' (5, a, 7, +2 more). Assigning"
            ' them random values. If you want to model new categories,'
            " please fit the data again using 'fit'."
        )

        # Assert
        with pytest.warns(UserWarning, match=warning_msg_1):
            transformed = transformer._transform(data_1)
        with pytest.warns(UserWarning, match=warning_msg_2):
            transformed = transformer._transform(data_2)

        assert transformed.iloc[4] >= 0
        assert transformed.iloc[4] < 1

    @patch('rdt.transformers.categorical.check_nan_in_transform')
    @patch('rdt.transformers.categorical.try_convert_to_dtype')
    def test__reverse_transform(self, mock_convert_dtype, mock_check_nan):
        """Test the ``_reverse_transform``."""
        # Setup
        data = pd.Series([1, 2, 3, 2, 2, 1, 3, 3, 2])
        transformer = UniformEncoder()
        transformer.dtype = np.int64
        transformer.frequencies = {1: 0.222222, 2: 0.444444, 3: 0.333333}
        transformer.intervals = {
            1: [0, 0.222222],
            2: [0.222222, 0.666666],
            3: [0.666666, 1.0],
        }

        transformed = pd.Series([
            0.12,
            0.254,
            0.789,
            0.43,
            0.56,
            0.08,
            0.67,
            0.98,
            0.36,
        ])
        mock_convert_dtype.return_value = pd.Series([
            1,
            2,
            3,
            2,
            2,
            1,
            3,
            3,
            2,
        ])

        # Run
        output = transformer._reverse_transform(transformed)

        # Asserts
        pd.testing.assert_series_equal(output, data)
        mock_input_data = mock_check_nan.call_args.args[0]
        mock_input_dtype = mock_check_nan.call_args.args[1]
        pd.testing.assert_series_equal(mock_input_data, transformed)
        assert mock_input_dtype == transformer.dtype
        mock_convert_dtype.assert_called_once()

    def test__reverse_transform_nans(self):
        """Test ``_reverse_transform`` for data with NaNs."""
        # Setup
        data = pd.Series([
            'a',
            'b',
            'NaN',
            np.nan,
            'NaN',
            'b',
            'b',
            'a',
            'b',
            np.nan,
        ])
        transformer = UniformEncoder()
        transformer.dtype = object
        transformer.frequencies = {'a': 0.2, 'b': 0.4, 'NaN': 0.2, np.nan: 0.2}
        transformer.intervals = {
            'a': [0, 0.2],
            'b': [0.2, 0.6],
            'NaN': [0.6, 0.8],
            np.nan: [0.8, 1],
        }

        transformed = pd.Series([
            0.12,
            0.254,
            0.789,
            0.88,
            0.69,
            0.53,
            0.47,
            0.08,
            0.39,
            0.92,
        ])

        # Run
        output = transformer._reverse_transform(transformed)

        # Asserts
        pd.testing.assert_series_equal(output, data)

    def test__reverse_transform_integer_and_nans(self):
        """Test the ``reverse_transform`` method with integers and nans.

        Test that the method correctly reverse transforms the data
        when the initial data is integers and the transformed data has nans.
        """
        # Setup
        transformer = UniformEncoder()
        transformer.frequencies = {11: 0.2, 12: 0.3, 13: 0.5}
        transformer.intervals = {11: [0, 0.2], 12: [0.2, 0.5], 13: [0.5, 1]}
        transformer.dtype = np.int64
        data = pd.Series([0.1, 0.25, np.nan, 0.65])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series([11, 12, np.nan, 13]))

    def test__reverse_transform_empty_intervals(self):
        """Test ``_reverse_transform``with nothing learned."""
        # Setup
        transformer = UniformEncoder(missing_value_encoding=None)
        transformer.intervals = {}
        transformer.dtype = 'object'
        data = pd.Series([0.1, np.nan], name='column_name')

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        expected = pd.Series([np.nan, np.nan], name='column_name')
        pd.testing.assert_series_equal(out, expected, check_dtype=False)


@pytest.fixture(autouse=True)
def _setup_caplog(caplog):
    """Define the logging for info message."""
    caplog.set_level(logging.INFO)


class TestOneHotEncoder:
    def test__prepare_data_empty_lists(self):
        # Setup
        ohe = OneHotEncoder()
        data = [[], [], []]

        # Assert
        with pytest.raises(ValueError, match='Unexpected format.'):
            ohe._prepare_data(data)

    def test__prepare_data_nested_lists(self):
        # Setup
        ohe = OneHotEncoder()
        data = [[[]]]

        # Assert
        with pytest.raises(ValueError, match='Unexpected format.'):
            ohe._prepare_data(data)

    def test__prepare_data_list_of_lists(self):
        # Setup
        ohe = OneHotEncoder()

        # Run
        data = [['a'], ['b'], ['c']]
        out = ohe._prepare_data(data)

        # Assert
        expected = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test__prepare_data_pandas_series(self):
        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        out = ohe._prepare_data(data)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test__fit_dummies_no_nans(self):
        """Test the ``_fit`` method without nans.

        Check that ``self.dummies`` does not
        contain nans.

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 2, 'c'])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 2, 'c'])
        assert ohe.dtype == 'object'

    def test__fit_dummies_nans(self):
        """Test the ``_fit`` method without nans.

        Check that ``self.dummies`` contain ``np.nan``.

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 2, 'c', None])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 2, 'c', np.nan])
        assert ohe.output_properties == {
            'value0': {'sdtype': 'float', 'next_transformer': None},
            'value1': {'sdtype': 'float', 'next_transformer': None},
            'value2': {'sdtype': 'float', 'next_transformer': None},
            'value3': {'sdtype': 'float', 'next_transformer': None},
        }

    def test__fit_no_nans(self):
        """Test the ``_fit`` method without nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be activated

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 'b', 'c'])
        np.testing.assert_array_equal(ohe._uniques, ['a', 'b', 'c'])
        assert ohe._dummy_encoded
        assert not ohe._dummy_na

    def test__fit_no_nans_numeric(self):
        """Test the ``_fit`` method without nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be deactivated

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series([1, 2, 3])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, [1, 2, 3])
        np.testing.assert_array_equal(ohe._uniques, [1, 2, 3])
        assert not ohe._dummy_encoded
        assert not ohe._dummy_na

    def test__fit_nans(self):
        """Test the ``_fit`` method with nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        and NA should be activated.

        Input:
        - Series with containing nan values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'b', None])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 'b', np.nan])
        np.testing.assert_array_equal(ohe._uniques, ['a', 'b'])
        assert ohe._dummy_encoded
        assert ohe._dummy_na

    def test__fit_nans_numeric(self):
        """Test the ``_fit`` method with nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be deactivated and NA activated.

        Input:
        - Series with containing nan values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series([1, 2, np.nan])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, [1, 2, np.nan])
        np.testing.assert_array_equal(ohe._uniques, [1, 2])
        assert not ohe._dummy_encoded
        assert ohe._dummy_na

    def test__fit_single(self):
        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a'])

    def test__transform_no_nan(self):
        """Test the ``_transform`` method without nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation.

        Input:
        - Series with values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._uniques = ['a', 'b', 'c']
        ohe._num_dummies = 3

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_no_nan_categorical(self):
        """Test the ``_transform`` method without nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        using the categorical branch.

        Input:
        - Series with categorical values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._uniques = ['a', 'b', 'c']
        ohe._indexer = [0, 1, 2]
        ohe._num_dummies = 3
        ohe._dummy_encoded = True

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans_encoded(self):
        """Test the ``_transform`` method with nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation. Null
        values should be represented by the same encoding.

        Input:
        - Series with values containing nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohe._uniques = ['a', 'b']
        ohe._dummy_na = True
        ohe._num_dummies = 2

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans_categorical(self):
        """Test the ``_transform`` method with nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation using
        the categorical branch. Null values should be
        represented by the same encoding.

        Input:
        - Series with categorical values containing nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohe._uniques = ['a', 'b']
        ohe._indexer = [0, 1]
        ohe._dummy_na = True
        ohe._num_dummies = 2
        ohe._dummy_encoded = True

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single_column(self):
        """Test the ``_transform`` with one category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        where it should be a single column.

        Input:
        - Series with a single category
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._uniques = ['a']
        ohe._num_dummies = 1

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([[1], [1], [1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single_categorical(self):
        """Test the ``_transform`` with one category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        using the categorical branch where it should
        be a single column.

        Input:
        - Series with a single category
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._uniques = ['a']
        ohe._indexer = [0]
        ohe._num_dummies = 1
        ohe._dummy_encoded = True

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([[1], [1], [1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_zeros(self):
        """Test the ``_transform`` with unknown category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        where it should be a column of zeros.

        Input:
        - Series with unknown values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        pd.Series(['a'])
        ohe._uniques = ['a']
        ohe._num_dummies = 1

        # Run
        out = ohe._transform_helper(pd.Series(['b', 'b', 'b']))

        # Assert
        expected = np.array([[0], [0], [0]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_zeros_categorical(self):
        """Test the ``_transform`` with unknown category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        using the categorical branch where it should
        be a column of zeros.

        Input:
        - Series with categorical and unknown values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        pd.Series(['a'])
        ohe._uniques = ['a']
        ohe._indexer = [0]
        ohe._num_dummies = 1
        ohe.dummy_encoded = True

        # Run
        out = ohe._transform_helper(pd.Series(['b', 'b', 'b']))

        # Assert
        expected = np.array([[0], [0], [0]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_unknown_nan(self):
        """Test the ``_transform`` with unknown and nans.

        This is an edge case for ``_transform`` where
        unknowns should be zeros and nans should be
        the last entry in the column.

        Input:
        - Series with unknown and nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        pd.Series(['a'])
        ohe._uniques = ['a']
        ohe._dummy_na = True
        ohe._num_dummies = 1

        # Run
        out = ohe._transform_helper(pd.Series(['b', 'b', np.nan]))

        # Assert
        expected = np.array([[0, 0], [0, 0], [0, 1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_no_nans(self):
        """Test the ``transform`` without nans.

        In this test ``transform`` should return an identity
        matrix representing each item in the input.

        Input:
        - Series with categorical values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._fit(data)

        # Run
        out = ohe._transform(data)

        # Assert
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans(self):
        """Test the ``transform`` with nans.

        In this test ``transform`` should return an identity matrix
        representing each item in the input as well as nans.

        Input:
        - Series with categorical values and nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', None])
        ohe._fit(data)

        # Run
        out = ohe._transform(data)

        # Assert
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single_column_filled_with_ones(self):
        """Test the ``transform`` on a single category.

        In this test ``transform`` should return a column
        filled with ones.

        Input:
        - Series with a single categorical value
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Run
        out = ohe._transform(data)

        # Assert
        expected = np.array([[1], [1], [1]])
        np.testing.assert_array_equal(out, expected)

    def test__transform_unknown(self):
        """Test the ``transform`` with unknown data.

        In this test ``transform`` should raise a warning due to the attempt
        of transforming data with previously unseen categories.

        Input:
        - Series with unknown categorical values
        Output:
        - one-hot encoding of the input, with the unseen category encoded as 0s
        """
        # Setup
        ohe = OneHotEncoder()
        fit_data = pd.Series([1, 2, 3, np.nan])
        ohe._fit(fit_data)

        # Run
        warning_msg = re.escape(
            'The data contains 1 new categories that were not '
            "seen in the original data (examples: {'4'}). Creating "
            'a vector of all 0s. If you want to model new categories, '
            'please fit the transformer again with the new data.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            transform_data = pd.Series([1, 2, np.nan, '4'], dtype='object')
            out = ohe._transform(transform_data)

        # Assert
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_numeric(self):
        """Test the ``transform`` on numeric input.

        In this test ``transform`` should return a matrix
        representing each item in the input as one-hot encodings.

        Input:
        - Series with numeric input
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series([1, 2])
        ohe._fit(data)

        expected = np.array([
            [1, 0],
            [0, 1],
        ])

        # Run
        out = ohe._transform(data)

        # Assert
        assert not ohe._dummy_encoded
        np.testing.assert_array_equal(out, expected)

    @patch('rdt.transformers.categorical.check_nan_in_transform')
    @patch('rdt.transformers.categorical.try_convert_to_dtype')
    def test__reverse_transform_no_nans(self, mock_convert_dtype, mock_check_nan):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._fit(data)
        mock_convert_dtype.return_value = data

        # Run
        transformed = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        pd.testing.assert_series_equal(out, expected)
        mock_input_data = mock_check_nan.call_args.args[0]
        mock_input_dtype = mock_check_nan.call_args.args[1]
        np.testing.assert_array_equal(mock_input_data, transformed)
        assert mock_input_dtype == 'O'
        mock_convert_dtype.assert_called_once()

    def test__reverse_transform_nans(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', None])
        ohe._fit(data)

        # Run
        transformed = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', None])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_single(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Run
        transformed = np.array([[1], [1], [1]])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_1d(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Run
        transformed = pd.Series([1, 1, 1])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)


class TestLabelEncoder:
    def test___init__(self):
        """Passed arguments must be stored as attributes."""
        # Run
        transformer = LabelEncoder(add_noise='add_noise_value')

        # Asserts
        assert transformer.add_noise == 'add_noise_value'

    def test___init___bad_missing_value_encoding(self):
        """Test that the ``__init__`` raises error if ``missing_value_encoding`` is invalid."""
        # Run / Assert
        message = (
            "'missing_value_encoding' must be one of the following values: None or 'new_category'."
        )
        with pytest.raises(TransformerInputError, match=message):
            LabelEncoder(missing_value_encoding='bad_value')

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that a unique integer representation for each category of the data is stored
        in the ``categories_to_values`` attribute, and the reverse is stored in the
        ``values_to_categories`` attribute .

        Setup:
            - create an instance of the ``LabelEncoder``.

        Input:
            - a pandas series.

        Side effects:
            - set the ``values_to_categories`` dictionary to the appropriate value.
            - set ``categories_to_values`` dictionary to the appropriate value.
        """
        # Setup
        data = pd.Series([1, 2, 3, 2, 1])
        transformer = LabelEncoder()

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.values_to_categories == {0: 1, 1: 2, 2: 3}
        assert transformer.categories_to_values == {1: 0, 2: 1, 3: 2}
        assert transformer.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
        }

    def test__fit_missing_value_encoding_none(self):
        """Test that missing values are ignored during fit when configured."""
        # Setup
        data = pd.Series(['foo', None, 'bar', np.nan])
        transformer = LabelEncoder(missing_value_encoding=None)

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.values_to_categories == {0: 'foo', 1: 'bar'}
        assert transformer.categories_to_values == {'foo': 0, 'bar': 1}

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that each category of the passed data is replaced with its corresponding
        integer value.

        Setup:
            - create an instance of the ``LabelEncoder``, where ``categories_to_values``
            and ``values_to_categories`` are set to dictionaries.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformer = LabelEncoder()
        transformer.categories_to_values = {1: 0, 2: 1, 3: 2}
        transformer.values_to_categories = {0: 1, 1: 2, 2: 3}

        # Run
        warning_msg = re.escape(
            'The data contains 1 new categories that were not '
            'seen in the original data (examples: {4}). Assigning '
            'them random values. If you want to model new categories, '
            'please fit the transformer again with the new data.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            transformed = transformer._transform(data)

        # Assert
        expected = pd.Series([0.0, 1.0, 2.0])
        pd.testing.assert_series_equal(transformed[:-1], expected)

        assert 0 <= transformed[3] <= 2

    def test__transform_missing_value_encoding_none(self):
        """Test missing values are not encoded during transform when configured."""
        # Setup
        data = pd.Series(['foo', None, 'bar', np.nan])
        transformer = LabelEncoder(missing_value_encoding=None)
        transformer.categories_to_values = {'foo': 0, 'bar': 1}
        transformer.values_to_categories = {0: 'foo', 1: 'bar'}

        # Run
        transformed = transformer._transform(data)

        # Assert
        expected = pd.Series([0.0, np.nan, 1.0, np.nan])
        pd.testing.assert_series_equal(transformed, expected)

    def test__transform_add_noise(self):
        """Test the ``_transform`` method with ``add_noise``.

        Validate that the method correctly transforms the categories when ``add_noise`` is True.

        Setup:
            - create an instance of the ``LabelEncoder``, where ``categories_to_values``
            and ``values_to_categories`` are set to dictionaries.
            - set ``add_noise`` to True.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformer = LabelEncoder(add_noise=True)
        transformer.categories_to_values = {1: 0, 2: 1, 3: 2}
        transformer.values_to_categories = {0: 1, 1: 2, 2: 3}

        # Run
        transformed = transformer._transform(data)

        # Assert
        assert 0 <= transformed[0] < 1
        assert 1 <= transformed[1] < 2
        assert 2 <= transformed[2] < 3
        assert 0 <= transformed[3] < 3

    def test__transform_unseen_categories(self):
        """Test the ``_transform`` method with multiple unseen categories.

        Validate that each category of the passed data is replaced with its corresponding
        integer value.

        Setup:
            - create an instance of the ``LabelEncoder``, where ``categories_to_values``
            and ``values_to_categories`` are set to dictionaries.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        fit_data = pd.Series(['a', 2, True])
        transformer = LabelEncoder()
        transformer.categories_to_values = {'a': 0, 2: 1, True: 2}
        transformer.values_to_categories = {0: 'a', 1: 2, 2: True}

        # Run
        with pytest.warns(UserWarning):
            transform_data = pd.Series([
                'a',
                2,
                True,
                np.nan,
                np.nan,
                np.nan,
                'b',
                False,
                3,
            ])
            transformed = transformer._transform(transform_data)

        # Assert
        expected = pd.Series([0.0, 1.0, 2.0])
        pd.testing.assert_series_equal(transformed[:3], expected)

        assert all(0 <= value < len(fit_data) for value in transformed[3:])

    def test__reverse_transform_clips_values(self):
        """Test the ``_reverse_transform`` method with values not in map.

        If a value that is not in ``values_to_categories`` is passed
        to ``reverse_transform``, then the value should be clipped to
        the range of the dict's keys.

        Input:
        - array with values outside of dict
        Output:
        - categories corresponding to closest key in the dict
        """
        # Setup
        transformer = LabelEncoder()
        transformer.values_to_categories = {0: 'a', 1: 'b', 2: 'c'}
        data = pd.Series([0, 1, 10])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series(['a', 'b', 'c']))

    def test__reverse_transform_empty_values_to_categories(self):
        """Test the ``_reverse_transform`` method when nothing was learned."""
        # Setup
        transformer = LabelEncoder(missing_value_encoding=None)
        transformer.values_to_categories = {}
        transformer.dtype = 'object'
        data = pd.Series([0.0, np.nan], name='column_name')

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        expected = pd.Series([np.nan, np.nan], name='column_name')
        pd.testing.assert_series_equal(out, expected, check_dtype=False)

    @patch('rdt.transformers.categorical.check_nan_in_transform')
    @patch('rdt.transformers.categorical.try_convert_to_dtype')
    def test__reverse_transform_add_noise(self, mock_convert_dtype, mock_check_nan):
        """Test the ``_reverse_transform`` method with ``add_noise``.

        Test that the method correctly reverse transforms the data
        when ``add_noise`` is set to True.

        Input:
            - pd.Series
        Output:
            - corresponding categories
        """
        # Setup
        transformer = LabelEncoder(add_noise=True)
        transformer.values_to_categories = {0: 'a', 1: 'b', 2: 'c'}
        data = pd.Series([0.5, 1.0, 10.9])
        mock_convert_dtype.return_value = pd.Series(['a', 'b', 'c'])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series(['a', 'b', 'c']))
        mock_input_data = mock_check_nan.call_args.args[0]
        mock_input_dtype = mock_check_nan.call_args.args[1]
        pd.testing.assert_series_equal(mock_input_data, data)
        assert mock_input_dtype == 'O'
        mock_convert_dtype.assert_called_once()

    def test__reverse_transform_integer_and_nans(self):
        """Test the ``reverse_transform`` method with integers and nans.

        Test that the method correctly reverse transforms the data
        when the initial data is integers and the transformed data has nans.
        """
        # Setup
        transformer = LabelEncoder()
        transformer.values_to_categories = {0: 11, 1: 12, 2: 13}
        transformer.dtype = 'int'
        data = pd.Series([0, 1, np.nan])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series([11, 12, np.nan]))
