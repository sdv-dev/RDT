import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.categorical import (
    CategoricalFuzzyTransformer, CategoricalTransformer, LabelEncodingTransformer,
    OneHotEncodingTransformer)

RE_SSN = re.compile(r'\d\d\d-\d\d-\d\d\d\d')


class TestCategoricalTransformer:

    def test___setstate__(self):
        """Test the ``__set_state__`` method.

        Validate that the ``__dict__`` attribute is correctly udpdated when

        Setup:
            - create an instance of a ``CategoricalTransformer``.

        Side effect:
            - it updates the ``__dict__`` attribute of the object.
        """
        # Setup
        transformer = CategoricalTransformer()

        # Run
        transformer.__setstate__({
            'intervals': {
                None: 'abc'
            }
        })

        # Assert
        assert transformer.__dict__['intervals'][np.nan] == 'abc'

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

    def test_is_transform_deterministic(self):
        """Test the ``is_transform_deterministic`` method.

        Validate that this method returs the opposite boolean value of the ``fuzzy`` parameter.

        Setup:
            - initialize a ``CategoricalTransformer`` with ``fuzzy = True``.

        Output:
            - the boolean value which is the opposite of ``fuzzy``.
        """
        # Setup
        transformer = CategoricalTransformer(fuzzy=True)

        # Run
        output = transformer.is_transform_deterministic()

        # Assert
        assert output is False

    def test_is_composition_identity(self):
        """Test the ``is_composition_identity`` method.

        Since ``COMPOSITION_IS_IDENTITY`` is True, just validates that the method
        returns the opposite boolean value of the ``fuzzy`` parameter.

        Setup:
            - initialize a ``CategoricalTransformer`` with ``fuzzy = True``.

        Output:
            - the boolean value which is the opposite of ``fuzzy``.
        """
        # Setup
        transformer = CategoricalTransformer(fuzzy=True)

        # Run
        output = transformer.is_composition_identity()

        # Assert
        assert output is False

    def test__get_intervals(self):
        """Test the ``_get_intervals`` method.

        Validate that the intervals for each categorical value are correct.

        Input:
            - a pandas series containing categorical values.

        Output:
            - a tuple, where the first element describes the intervals for each
            categorical value (start, end).
        """
        # Run
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        result = CategoricalTransformer._get_intervals(data)

        # Asserts
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            'bar': (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }
        expected_means = pd.Series({
            'foo': 0.25,
            'bar': 0.6666666666666666,
            'tar': 0.9166666666666666
        })
        expected_starts = pd.DataFrame({
            'category': ['foo', 'bar', 'tar'],
            'start': [0, 0.5, 0.8333333333333333]
        }).set_index('start')

        assert result[0] == expected_intervals
        pd.testing.assert_series_equal(result[1], expected_means)
        pd.testing.assert_frame_equal(result[2], expected_starts)

    def test__get_intervals_nans(self):
        """Test the ``_get_intervals`` method when data contains nan's.

        Validate that the intervals for each categorical value are correct, when passed
        data containing nan values.

        Input:
            - a pandas series cotaining nan values and categorical values.

        Output:
            - a tuple, where the first element describes the intervals for each
            categorical value (start, end).
        """
        # Setup
        data = pd.Series(['foo', np.nan, None, 'foo', 'foo', 'tar'])

        # Run
        result = CategoricalTransformer._get_intervals(data)

        # Assert
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            np.nan: (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }
        expected_means = pd.Series({
            'foo': 0.25,
            np.nan: 0.6666666666666666,
            'tar': 0.9166666666666666
        })
        expected_starts = pd.DataFrame({
            'category': ['foo', np.nan, 'tar'],
            'start': [0, 0.5, 0.8333333333333333]
        }).set_index('start')

        assert result[0] == expected_intervals
        pd.testing.assert_series_equal(result[1], expected_means)
        pd.testing.assert_frame_equal(result[2], expected_starts)

    def test__fit_intervals(self):
        # Setup
        transformer = CategoricalTransformer()

        # Run
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        transformer._fit(data)

        # Asserts
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            'bar': (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }
        expected_means = pd.Series({
            'foo': 0.25,
            'bar': 0.6666666666666666,
            'tar': 0.9166666666666666
        })
        expected_starts = pd.DataFrame({
            'category': ['foo', 'bar', 'tar'],
            'start': [0, 0.5, 0.8333333333333333]
        }).set_index('start')

        assert transformer.intervals == expected_intervals
        pd.testing.assert_series_equal(transformer.means, expected_means)
        pd.testing.assert_frame_equal(transformer.starts, expected_starts)

    def test__get_value_no_fuzzy(self):
        # Setup
        transformer = CategoricalTransformer(fuzzy=False)
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            np.nan: (0.5, 1.0, 0.75, 0.5 / 6),
        }

        # Run
        result_foo = transformer._get_value('foo')
        result_nan = transformer._get_value(np.nan)

        # Asserts
        assert result_foo == 0.25
        assert result_nan == 0.75

    @patch('rdt.transformers.categorical.norm')
    def test__get_value_fuzzy(self, norm_mock):
        # setup
        norm_mock.rvs.return_value = 0.2745

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

    def test__reverse_transform_array(self):
        """Test reverse_transform a numpy.array"""
        # Setup
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        rt_data = np.array([-0.6, 0.5, 0.6, 0.2, 0.1, -0.2])
        transformer = CategoricalTransformer()

        # Run
        transformer._fit(data)
        result = transformer._reverse_transform(rt_data)

        # Asserts
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            'bar': (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
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
            - data with 5 rows.

        Output:
            - the output of `_transform_by_category`.

        Side effects:
            - `_transform_by_category` will be called once.
        """
        # Setup
        data = pd.Series([1, 3, 3, 2, 1])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        # Run
        transformed = CategoricalTransformer._transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._transform_by_category.assert_called_once_with(data)
        assert transformed == categorical_transformer_mock._transform_by_category.return_value

    def test__transform_by_category(self):
        """Test the `_transform_by_category` method with numerical data.

        Expect that the correct transformed data is returned.

        Setup:
            The categorical transformer is instantiated with 4 categories and intervals.

        Input:
            - data with 5 rows.

        Ouptut:
            - the transformed data.
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

    def test__transform_by_category_nans(self):
        """Test the ``_transform_by_category`` method with data containing nans.

        Validate that the data is transformed correctly when it contains nan's.

        Setup:
            - the categorical transformer is instantiated, and the appropriate ``intervals``
            attribute is set.

        Input:
            - a pandas series containing nan's.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([np.nan, 3, 3, 2, np.nan])
        transformer = CategoricalTransformer()
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            np.nan: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_category(data)

        # Asserts
        expected = np.array([0.875, 0.375, 0.375, 0.625, 0.875])
        assert (transformed == expected).all()

    @patch('rdt.transformers.categorical.norm')
    def test__transform_by_category_fuzzy_true(self, norm_mock):
        """Test the ``_transform_by_category`` method when ``fuzzy`` is True.

        Validate that the data is transformed correctly when ``fuzzy`` is True.

        Setup:
            - the categorical transformer is instantiated with ``fuzzy`` as True,
            and the appropriate ``intervals`` attribute is set.
            - the ``intervals`` attribute is set to a a dictionary of intervals corresponding
            to the elements of the passed data.
            - set the ``side_effect`` of the ``rvs_mock`` to the appropriate function.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.

        Side effect:
            - ``rvs_mock`` should be called four times, one for each element of the
            intervals dictionary.
        """
        # Setup
        def rvs_mock_func(loc, scale, **kwargs):
            return loc

        norm_mock.rvs.side_effect = rvs_mock_func

        data = pd.Series([1, 3, 3, 2, 1])
        transformer = CategoricalTransformer(fuzzy=True)
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_category(data)

        # Assert
        expected = np.array([0.875, 0.375, 0.375, 0.625, 0.875])
        assert (transformed == expected).all()
        norm_mock.rvs.assert_has_calls([
            call(0.125, 0.041666666666666664, size=0),
            call(0.375, 0.041666666666666664, size=2),
            call(0.625, 0.041666666666666664, size=1),
            call(0.875, 0.041666666666666664, size=2),
        ])

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
        transformed = CategoricalTransformer._transform(categorical_transformer_mock, data)

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
    def test__reverse_transform_by_matrix_called(self, psutil_mock):
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
        reverse = CategoricalTransformer._reverse_transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._reverse_transform_by_matrix.assert_called_once_with(data)
        assert reverse == categorical_transformer_mock._reverse_transform_by_matrix.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_matrix(self, psutil_mock):
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
        reverse = CategoricalTransformer._reverse_transform(
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

    def test__get_category_from_start(self):
        """Test the ``_get_category_from_start`` method.

        Setup:
            - instantiate a ``CategoricalTransformer``, and set the attribute ``starts``
            to a pandas dataframe with ``set_index`` as ``'start'``.

        Input:
            - an integer, an index from data.

        Output:
            - a category from the data.
        """
        # Setup
        transformer = CategoricalTransformer()
        transformer.starts = pd.DataFrame({
            'start': [0.0, 0.5, 0.7],
            'category': ['a', 'b', 'c']
        }).set_index('start')

        # Run
        category = transformer._get_category_from_start(2)

        # Assert
        assert category == 'c'

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
        reverse = CategoricalTransformer._reverse_transform(categorical_transformer_mock, data)

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
        reverse = transformer._reverse_transform(transformed)

        # Assert
        pd.testing.assert_series_equal(data, reverse)


class TestOneHotEncodingTransformer:

    def test___init__(self):
        """Test the ``__init__`` method.

        Validate that the passed arguments are stored as attributes.

        Input:
            - a string passed to the ``error_on_unknown`` parameter.

        Side effect:
            - the ``error_on_unknown`` attribute is set to the passed string.
        """
        # Run
        transformer = OneHotEncodingTransformer(error_on_unknown='error_value')

        # Asserts
        assert transformer.error_on_unknown == 'error_value'

    def test__prepare_data_empty_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = [[], [], []]

        # Assert
        with pytest.raises(ValueError, match='Unexpected format.'):
            ohet._prepare_data(data)

    def test__prepare_data_nested_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = [[[]]]

        # Assert
        with pytest.raises(ValueError, match='Unexpected format.'):
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

    def test_get_output_types(self):
        """Test the ``get_output_types`` method.

        Validate that the ``_add_prefix`` method is properly applied to the ``output_types``
        dictionary. For this class, the ``output_types`` dictionary is described as:

        {
            'value1': 'float',
            'value2': 'float',
            ...
        }

        The number of items in the dictionary is defined by the ``dummies`` attribute.

        Setup:
            - initialize a ``OneHotEncodingTransformer`` and set:
                - the ``dummies`` attribute to a list.
                - the ``column_prefix`` attribute to a string.

        Output:
            - the ``output_types`` dictionary, but with ``self.column_prefix``
            added to the beginning of the keys of the ``output_types`` dictionary.
        """
        # Setup
        transformer = OneHotEncodingTransformer()
        transformer.column_prefix = 'abc'
        transformer.dummies = [1, 2]

        # Run
        output = transformer.get_output_types()

        # Assert
        expected = {
            'abc.value0': 'float',
            'abc.value1': 'float'
        }
        assert output == expected

    def test__fit_dummies_no_nans(self):
        """Test the ``_fit`` method without nans.

        Check that ``self.dummies`` does not
        contain nans.

        Input:
        - Series with values
        """

        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 2, 'c'])
        ohet._fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 2, 'c'])

    def test__fit_dummies_nans(self):
        """Test the ``_fit`` method without nans.

        Check that ``self.dummies`` contain ``np.nan``.

        Input:
        - Series with values
        """

        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 2, 'c', None])
        ohet._fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 2, 'c', np.nan])

    def test__fit_no_nans(self):
        """Test the ``_fit`` method without nans.

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
        ohet._fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 'b', 'c'])
        np.testing.assert_array_equal(ohet._uniques, ['a', 'b', 'c'])
        assert ohet._dummy_encoded
        assert not ohet._dummy_na

    def test__fit_no_nans_numeric(self):
        """Test the ``_fit`` method without nans.

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
        ohet._fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, [1, 2, 3])
        np.testing.assert_array_equal(ohet._uniques, [1, 2, 3])
        assert not ohet._dummy_encoded
        assert not ohet._dummy_na

    def test__fit_nans(self):
        """Test the ``_fit`` method with nans.

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
        ohet._fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 'b', np.nan])
        np.testing.assert_array_equal(ohet._uniques, ['a', 'b'])
        assert ohet._dummy_encoded
        assert ohet._dummy_na

    def test__fit_nans_numeric(self):
        """Test the ``_fit`` method with nans.

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
        ohet._fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, [1, 2, np.nan])
        np.testing.assert_array_equal(ohet._uniques, [1, 2])
        assert not ohet._dummy_encoded
        assert ohet._dummy_na

    def test__fit_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'a', 'a'])
        ohet._fit(data)

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
        ohet._uniques = ['a', 'b', 'c']
        ohet._num_dummies = 3

        # Run
        out = ohet._transform_helper(data)

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
        ohet._uniques = ['a', 'b', 'c']
        ohet._indexer = [0, 1, 2]
        ohet._num_dummies = 3
        ohet._dummy_encoded = True

        # Run
        out = ohet._transform_helper(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohet._uniques = ['a', 'b']
        ohet._dummy_na = True
        ohet._num_dummies = 2

        # Run
        out = ohet._transform_helper(data)

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
        ohet._uniques = ['a', 'b']
        ohet._indexer = [0, 1]
        ohet._dummy_na = True
        ohet._num_dummies = 2
        ohet._dummy_encoded = True

        # Run
        out = ohet._transform_helper(data)

        # Assert
        expected = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet._uniques = ['a']
        ohet._num_dummies = 1

        # Run
        out = ohet._transform_helper(data)

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
        ohet._uniques = ['a']
        ohet._indexer = [0]
        ohet._num_dummies = 1
        ohet._dummy_encoded = True

        # Run
        out = ohet._transform_helper(data)

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
        ohet._uniques = ['a']
        ohet._num_dummies = 1

        # Run
        out = ohet._transform_helper(pd.Series(['b', 'b', 'b']))

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
        ohet._uniques = ['a']
        ohet._indexer = [0]
        ohet._num_dummies = 1
        ohet.dummy_encoded = True

        # Run
        out = ohet._transform_helper(pd.Series(['b', 'b', 'b']))

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
        ohet._uniques = ['a']
        ohet._dummy_na = True
        ohet._num_dummies = 1

        # Run
        out = ohet._transform_helper(pd.Series(['b', 'b', np.nan]))

        # Assert
        expected = np.array([
            [0, 0],
            [0, 0],
            [0, 1]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet._fit(data)

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
        ohet._fit(data)

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
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
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet._fit(data)

        # Run
        out = ohet._transform(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_unknown(self):
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
        ohet._fit(data)

        # Assert
        with np.testing.assert_raises(ValueError):
            ohet._transform(['b'])

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
        ohet = OneHotEncodingTransformer()
        data = pd.Series([1, 2])
        ohet._fit(data)

        expected = np.array([
            [1, 0],
            [0, 1],
        ])

        # Run
        out = ohet._transform(data)

        # Assert
        assert not ohet._dummy_encoded
        np.testing.assert_array_equal(out, expected)

    def test__reverse_transform_no_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet._fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohet._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', None])
        ohet._fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohet._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', None])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet._fit(data)

        # Run
        transformed = np.array([
            [1],
            [1],
            [1]
        ])
        out = ohet._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_1d(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet._fit(data)

        # Run
        transformed = pd.Series([1, 1, 1])
        out = ohet._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)


class TestLabelEncodingTransformer:

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that a unique integer representation for each category of the data is stored
        in the ``categories_to_values`` attribute, and the reverse is stored in the
        ``values_to_categories`` attribute .

        Setup:
            - create an instance of the ``LabelEncodingTransformer``.

        Input:
            - a pandas series.

        Side effects:
            - set the ``values_to_categories`` dictionary to the appropriate value.
            - set ``categories_to_values`` dictionary to the appropriate value.
        """
        # Setup
        data = pd.Series([1, 2, 3, 2, 1])
        transformer = LabelEncodingTransformer()

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.values_to_categories == {0: 1, 1: 2, 2: 3}
        assert transformer.categories_to_values == {1: 0, 2: 1, 3: 2}

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that each category of the passed data is replaced with its corresponding
        integer value.

        Setup:
            - create an instance of the ``LabelEncodingTransformer``, where
            ``categories_to_values`` is set to a dictionary.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([1, 2, 3])
        transformer = LabelEncodingTransformer()
        transformer.categories_to_values = {1: 0, 2: 1, 3: 2}

        # Run
        transformed = transformer._transform(data)

        # Assert
        pd.testing.assert_series_equal(transformed, pd.Series([0, 1, 2]))

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
        transformer = LabelEncodingTransformer()
        transformer.values_to_categories = {0: 'a', 1: 'b', 2: 'c'}
        data = pd.Series([0, 1, 10])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series(['a', 'b', 'c']))


class TestCategoricalFuzzyTransformer:

    def test___init__(self):
        """Test that the ``__init__`` method uses ``fuzzy==True`` by default."""
        # Setup
        transformer = CategoricalFuzzyTransformer()

        # Assert
        assert transformer.fuzzy
