import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import (
    CategoricalTransformer, LabelEncodingTransformer, OneHotEncodingTransformer)

RE_SSN = re.compile(r'\d\d\d-\d\d-\d\d\d\d')


class TestCategoricalTransformer:

    def test___init__(self):
        """Passed arguments must be stored as attributes."""
        # Run
        transformer = CategoricalTransformer(
            fuzzy='fuzzy_value',
            clip='clip_value',
        )

        # Asserts
        assert transformer.fuzzy == 'fuzzy_value'
        assert transformer.clip == 'clip_value'

    def test__get_intervals(self):
        # Run
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        result = CategoricalTransformer._get_intervals(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'bar': (0.5, 0.8333333333333333, 0.6666666666666666, 0.05555555555555555),
            'tar': (0.8333333333333333, 0.9999999999999999, 0.9166666666666666,
                    0.027777777777777776)
        }
        assert result[0] == expected_intervals

    def test_fit(self):
        # Setup
        transformer = CategoricalTransformer()

        # Run
        data = np.array(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        transformer.fit(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'bar': (0.5, 0.8333333333333333, 0.6666666666666666, 0.05555555555555555),
            'tar': (0.8333333333333333, 0.9999999999999999, 0.9166666666666666,
                    0.027777777777777776)
        }
        assert transformer.intervals == expected_intervals

    def test__get_value_no_fuzzy(self):
        # Setup
        transformer = CategoricalTransformer(fuzzy=False)
        transformer.fuzzy = False
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        # Run
        result = transformer._get_value('foo')

        # Asserts
        assert result == 0.25

    @patch('scipy.stats.norm.rvs')
    def test__get_value_fuzzy(self, rvs_mock):
        # setup
        rvs_mock.return_value = 0.2745

        transformer = CategoricalTransformer(fuzzy=True)
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        # Run
        result = transformer._get_value('foo')

        # Asserts
        assert result == 0.2745

    def test__normalize_no_clip(self):
        """Test normalize data"""
        # Setup
        transformer = CategoricalTransformer(clip=False)

        # Run
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])
        result = transformer._normalize(data)

        # Asserts
        expect = pd.Series([0.57, 0.1234, 0.5, 0.69], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test__normalize_clip(self):
        """Test normalize data with clip=True"""
        # Setup
        transformer = CategoricalTransformer(clip=True)

        # Run
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])
        result = transformer._normalize(data)

        # Asserts
        expect = pd.Series([0.0, 0.1234, 1.0, 0.0], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test_reverse_transform_array(self):
        """Test reverse_transform a numpy.array"""
        # Setup
        data = np.array(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        rt_data = np.array([-0.6, 0.5, 0.6, 0.2, 0.1, -0.2])
        transformer = CategoricalTransformer()

        # Run
        transformer.fit(data)
        result = transformer.reverse_transform(rt_data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'bar': (0.5, 0.8333333333333333, 0.6666666666666666, 0.05555555555555555),
            'tar': (0.8333333333333333, 0.9999999999999999, 0.9166666666666666,
                    0.027777777777777776)
        }
        assert transformer.intervals == expected_intervals

        expect = pd.Series(data)
        pd.testing.assert_series_equal(result, expect)

    def test__transform_by_category_called(self):
        """Test that the `_transform_by_category` method is called.

        When the number of rows is greater than the number of categories, expect
        that the `_transform_by_category` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories.
        Input:
            - data with 5 rows
        Output:
            - the output of `_transform_by_category`
        Side effects:
            - `_transform_by_category` will be called once
        """
        # Setup
        data = pd.Series([1, 3, 3, 2, 1])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        # Run
        transformed = CategoricalTransformer.transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._transform_by_category.assert_called_once_with(data)
        assert transformed == categorical_transformer_mock._transform_by_category.return_value

    def test__transform_by_category(self):
        """Test the `_transform_by_category` method with numerical data.

        Expect that the correct transformed data is returned.

        Setup:
            The categorical transformer is instantiated with 4 categories and intervals.
        Input:
            - data with 5 rows
        Ouptut:
            - the transformed data
        """
        # Setup
        data = pd.Series([1, 3, 3, 2, 1])
        transformer = CategoricalTransformer()
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_category(data)

        # Asserts
        expected = np.array([0.875, 0.375, 0.375, 0.625, 0.875])
        assert (transformed == expected).all()

    def test__transform_by_row_called(self):
        """Test that the `_transform_by_row` method is called.

        When the number of rows is less than or equal to the number of categories,
        expect that the `_transform_by_row` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories.
        Input:
            - data with 4 rows
        Output:
            - the output of `_transform_by_row`
        Side effects:
            - `_transform_by_row` will be called once
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        # Run
        transformed = CategoricalTransformer.transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._transform_by_row.assert_called_once_with(data)
        assert transformed == categorical_transformer_mock._transform_by_row.return_value

    def test__transform_by_row(self):
        """Test the `_transform_by_row` method with numerical data.

        Expect that the correct transformed data is returned.

        Setup:
            The categorical transformer is instantiated with 4 categories and intervals.
        Input:
            - data with 4 rows
        Ouptut:
            - the transformed data
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformer = CategoricalTransformer()
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_row(data)

        # Asserts
        expected = np.array([0.875, 0.625, 0.375, 0.125])
        assert (transformed == expected).all()

    @patch('psutil.virtual_memory')
    def test__reverse_transfrom_by_matrix_called(self, psutil_mock):
        """Test that the `_reverse_transform_by_matrix` method is called.

        When there is enough virtual memory, expect that the
        `_reverse_transform_by_matrix` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories. Also patch the
            `psutil.virtual_memory` function to return a large enough `available_memory`.
        Input:
            - numerical data with 4 rows
        Output:
            - the output of `_reverse_transform_by_matrix`
        Side effects:
            - `_reverse_transform_by_matrix` will be called once
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])
        categorical_transformer_mock._normalize.return_value = data

        virtual_memory = Mock()
        virtual_memory.available = 4 * 4 * 8 * 3 + 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = CategoricalTransformer.reverse_transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._reverse_transform_by_matrix.assert_called_once_with(data)
        assert reverse == categorical_transformer_mock._reverse_transform_by_matrix.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transfrom_by_matrix(self, psutil_mock):
        """Test the _reverse_transform_by_matrix method with numerical data

        Expect that the transformed data is correctly reverse transformed.

        Setup:
            The categorical transformer is instantiated with 4 categories and means. Also patch
            the `psutil.virtual_memory` function to return a large enough `available_memory`.
        Input:
            - transformed data with 4 rows
        Ouptut:
            - the original data
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformed = pd.Series([0.875, 0.625, 0.375, 0.125])

        transformer = CategoricalTransformer()
        transformer.means = pd.Series([0.125, 0.375, 0.625, 0.875], index=[4, 3, 2, 1])
        transformer.dtype = data.dtype

        virtual_memory = Mock()
        virtual_memory.available = 4 * 4 * 8 * 3 + 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = transformer._reverse_transform_by_matrix(transformed)

        # Assert
        pd.testing.assert_series_equal(data, reverse)

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_category_called(self, psutil_mock):
        """Test that the `_reverse_transform_by_category` method is called.

        When there is not enough virtual memory and the number of rows is greater than the
        number of categories, expect that the `_reverse_transform_by_category` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories. Also patch the
            `psutil.virtual_memory` function to return an `available_memory` of 1.
        Input:
            - numerical data with 5 rows
        Output:
            - the output of `_reverse_transform_by_category`
        Side effects:
            - `_reverse_transform_by_category` will be called once
        """
        # Setup
        transform_data = pd.Series([1, 3, 3, 2, 1])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])
        categorical_transformer_mock._normalize.return_value = transform_data

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = CategoricalTransformer.reverse_transform(
            categorical_transformer_mock, transform_data)

        # Asserts
        categorical_transformer_mock._reverse_transform_by_category.assert_called_once_with(
            transform_data)
        assert reverse == categorical_transformer_mock._reverse_transform_by_category.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_category(self, psutil_mock):
        """Test the _reverse_transform_by_category method with numerical data.

        Expect that the transformed data is correctly reverse transformed.

        Setup:
            The categorical transformer is instantiated with 4 categories, and the means
            and intervals are set for those categories. Also patch the `psutil.virtual_memory`
            function to return an `available_memory` of 1.
        Input:
            - transformed data with 5 rows
        Ouptut:
            - the original data
        """
        data = pd.Series([1, 3, 3, 2, 1])
        transformed = pd.Series([0.875, 0.375, 0.375, 0.625, 0.875])

        transformer = CategoricalTransformer()
        transformer.means = pd.Series([0.125, 0.375, 0.625, 0.875], index=[4, 3, 2, 1])
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }
        transformer.dtype = data.dtype

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        reverse = transformer._reverse_transform_by_category(transformed)

        pd.testing.assert_series_equal(data, reverse)

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_row_called(self, psutil_mock):
        """Test that the `_reverse_transform_by_row` method is called.

        When there is not enough virtual memory and the number of rows is less than or equal
        to the number of categories, expect that the `_reverse_transform_by_row` method
        is called.

        Setup:
            The categorical transformer is instantiated with 4 categories. Also patch the
            `psutil.virtual_memory` function to return an `available_memory` of 1.
        Input:
            - numerical data with 4 rows
        Output:
            - the output of `_reverse_transform_by_row`
        Side effects:
            - `_reverse_transform_by_row` will be called once
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])
        categorical_transformer_mock.starts = pd.DataFrame(
            [0., 0.25, 0.5, 0.75], index=[4, 3, 2, 1], columns=['category'])
        categorical_transformer_mock._normalize.return_value = data

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = CategoricalTransformer.reverse_transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._reverse_transform_by_row.assert_called_once_with(data)
        assert reverse == categorical_transformer_mock._reverse_transform_by_row.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_row(self, psutil_mock):
        """Test the _reverse_transform_by_row method with numerical data.

        Expect that the transformed data is correctly reverse transformed.

        Setup:
            The categorical transformer is instantiated with 4 categories, and the means, starts,
            and intervals are set for those categories. Also patch the `psutil.virtual_memory`
            function to return an `available_memory` of 1.
        Input:
            - transformed data with 4 rows
        Ouptut:
            - the original data
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformed = pd.Series([0.875, 0.625, 0.375, 0.125])

        transformer = CategoricalTransformer()
        transformer.means = pd.Series([0.125, 0.375, 0.625, 0.875], index=[4, 3, 2, 1])
        transformer.starts = pd.DataFrame(
            [4, 3, 2, 1], index=[0., 0.25, 0.5, 0.75], columns=['category'])
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }
        transformer.dtype = data.dtype

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_series_equal(data, reverse)


class TestOneHotEncodingTransformer:

    def test__prepare_data_empty_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = [[], [], []]

        # Assert
        with pytest.raises(ValueError):
            ohet._prepare_data(data)

    def test__prepare_data_nested_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = [[[]]]

        # Assert
        with pytest.raises(ValueError):
            ohet._prepare_data(data)

    def test__prepare_data_list_of_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = [['a'], ['b'], ['c']]
        out = ohet._prepare_data(data)

        # Assert
        expected = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test__prepare_data_pandas_series(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        out = ohet._prepare_data(data)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test_fit_no_nans(self):
        """Test the ``fit`` method without nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be activated

        Input:
        - Series with values
        """

        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 'b', 'c'])
        np.testing.assert_array_equal(ohet.decoder, ['a', 'b', 'c'])
        assert ohet.dummy_encoded
        assert not ohet.dummy_na

    def test_fit_no_nans_numeric(self):
        """Test the ``fit`` method without nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be deactivated

        Input:
        - Series with values
        """

        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series([1, 2, 3])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, [1, 2, 3])
        np.testing.assert_array_equal(ohet.decoder, [1, 2, 3])
        assert not ohet.dummy_encoded
        assert not ohet.dummy_na

    def test_fit_nans(self):
        """Test the ``fit`` method with nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        and NA should be activated.

        Input:
        - Series with containing nan values
        """

        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'b', None])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 'b'])
        np.testing.assert_array_equal(ohet.decoder, ['a', 'b', np.nan])
        assert ohet.dummy_encoded
        assert ohet.dummy_na

    def test_fit_nans_numeric(self):
        """Test the ``fit`` method with nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be deactivated and NA activated.

        Input:
        - Series with containing nan values
        """

        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series([1, 2, np.nan])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, [1, 2])
        np.testing.assert_array_equal(ohet.decoder, [1, 2, np.nan])
        assert not ohet.dummy_encoded
        assert ohet.dummy_na

    def test_fit_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a'])

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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet.dummies = ['a', 'b', 'c']
        ohet.num_dummies = 3

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet.dummies = ['a', 'b', 'c']
        ohet.indexer = [0, 1, 2]
        ohet.num_dummies = 3
        ohet.dummy_encoded = True

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans(self):
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohet.dummies = ['a', 'b']
        ohet.dummy_na = True
        ohet.num_dummies = 2

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohet.dummies = ['a', 'b']
        ohet.indexer = [0, 1]
        ohet.dummy_na = True
        ohet.num_dummies = 2
        ohet.dummy_encoded = True

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single(self):
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.dummies = ['a']
        ohet.num_dummies = 1

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.dummies = ['a']
        ohet.indexer = [0]
        ohet.num_dummies = 1
        ohet.dummy_encoded = True

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
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
        ohet = OneHotEncodingTransformer()
        pd.Series(['a'])
        ohet.dummies = ['a']
        ohet.num_dummies = 1

        # Run
        out = ohet._transform(pd.Series(['b', 'b', 'b']))

        # Assert
        expected = np.array([
            [0],
            [0],
            [0]
        ])
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
        ohet = OneHotEncodingTransformer()
        pd.Series(['a'])
        ohet.dummies = ['a']
        ohet.indexer = [0]
        ohet.num_dummies = 1
        ohet.dummy_encoded = True

        # Run
        out = ohet._transform(pd.Series(['b', 'b', 'b']))

        # Assert
        expected = np.array([
            [0],
            [0],
            [0]
        ])
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
        ohet = OneHotEncodingTransformer()
        pd.Series(['a'])
        ohet.dummies = ['a']
        ohet.dummy_na = True
        ohet.num_dummies = 1

        # Run
        out = ohet._transform(pd.Series(['b', 'b', np.nan]))

        # Assert
        expected = np.array([
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_no_nans(self):
        """Test the ``transform`` without nans.

        In this test ``transform`` should return an identity
        matrix representing each item in the input.

        Input:
        - Series with categorical values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet.fit(data)

        # Run
        out = ohet.transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_nans(self):
        """Test the ``transform`` with nans.

        In this test ``transform`` should return an identity matrix
        representing each item in the input as well as nans.

        Input:
        - Series with categorical values and nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', None])
        ohet.fit(data)

        # Run
        out = ohet.transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_single(self):
        """Test the ``transform`` on a single category.

        In this test ``transform`` should return a column
        filled with ones.

        Input:
        - Series with a single categorical value
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Run
        out = ohet.transform(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_unknown(self):
        """Test the ``transform`` with unknown data.

        In this test ``transform`` should raise an error
        due to the attempt of transforming data with previously
        unseen categories.

        Input:
        - Series with unknown categorical values
        """
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a'])
        ohet.fit(data)

        # Assert
        with np.testing.assert_raises(ValueError):
            ohet.transform(['b'])

    def test_transform_numeric(self):
        """Test the ``transform`` on numeric input.

        In this test ``transform`` should return a matrix
        representing each item in the input as one-hot encodings.

        Input:
        - Series with numeric input
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series([1, 2])
        ohet.fit(data)

        expected = np.array([
            [1, 0],
            [0, 1],
        ])

        # Run
        out = ohet.transform(data)

        # Assert
        assert not ohet.dummy_encoded
        np.testing.assert_array_equal(out, expected)

    def test_reverse_transform_no_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet.fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        pd.testing.assert_series_equal(out, expected)

    def test_reverse_transform_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', None])
        ohet.fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', None])
        pd.testing.assert_series_equal(out, expected)

    def test_reverse_transform_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Run
        transformed = np.array([
            [1],
            [1],
            [1]
        ])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)

    def test_reverse_transform_1d(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Run
        transformed = np.array([1, 1, 1])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)


class TestLabelEncodingTransformer:

    def test_reverse_transform_clips_values(self):
        """Test the ``reverse_transform`` method with values not in map.

        If a value that is not in ``values_to_categories`` is passed
        to ``reverse_transform``, then the value should be clipped to
        the range of the dict's keys.

        Input:
        - array with values outside of dict
        Output:
        - categories corresponding to closest key in the dict
        """
        # Setup
        transformer = LabelEncodingTransformer()
        transformer.values_to_categories = {0: 'a', 1: 'b', 2: 'c'}
        data = pd.Series([0, 1, 10])

        # Run
        out = transformer.reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series(['a', 'b', 'c']))
