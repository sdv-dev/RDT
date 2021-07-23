import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import CategoricalTransformer, OneHotEncodingTransformer

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
        data = pd.Series(['bar', 'foo', 'foo', 'tar'])
        result = CategoricalTransformer._get_intervals(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'tar': (0.5, 0.75, 0.625, 0.25 / 6),
            'bar': (0.75, 1, 0.875, 0.25 / 6)
        }
        assert result == expected_intervals

    def test_fit(self):
        # Setup
        transformer = CategoricalTransformer()

        # Run
        data = np.array(['bar', 'foo', 'foo', 'tar'])
        transformer.fit(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'tar': (0.5, 0.75, 0.625, 0.25 / 6),
            'bar': (0.75, 1, 0.875, 0.25 / 6)
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
        transformer = CategoricalTransformer()
        transformer.dtype = object
        transformer.intervals = {
            'foo': (0, 0.5),
            'bar': (0.5, 0.75),
            'tar': (0.75, 1),
        }

        # Run
        data = np.array([-0.6, 0.2, 0.6, -0.2])
        result = transformer.reverse_transform(data)

        # Asserts
        expect = pd.Series(['foo', 'foo', 'bar', 'tar'])

        pd.testing.assert_series_equal(result, expect)

    def test_reversible_strings(self):
        data = pd.Series(['a', 'b', 'a', 'c'])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_strings_2_categories(self):
        data = pd.Series(['a', 'b', 'a', 'b'])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_integers(self):
        data = pd.Series([1, 2, 3, 2])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_bool(self):
        data = pd.Series([True, False, True, False])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_mixed(self):
        data = pd.Series([True, 'a', 1, None])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

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
        """Test the ``_transform`` method without nans

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
        ohet.dummy_na = False
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

    def test__transform_nans(self):
        """Test the ``_transform`` method with nans

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
        ohet.dummy_na = False
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

    def test_transform_categorical(self):
        """Test the ``transform`` on categorical input.

        In this test ``transform`` should return a matrix
        representing each item in the input as one-hot encodings.

        Input:
        - Series with categorical input
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['1', '2'])
        ohet.fit(data)

        expected = np.array([
            [1, 0],
            [0, 1],
        ])

        # Run
        out = ohet.transform(data)

        # Assert
        assert ohet.dummy_encoded
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
