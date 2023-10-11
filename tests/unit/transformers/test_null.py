"""Unit tests for the NullTransformer."""

import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt.errors import TransformerInputError
from rdt.transformers import NullTransformer


class TestNullTransformer:

    def test___init__default(self):
        """Test the initialization without passing any default arguments.

        When no arguments are passed, the attributes should be populated
        with the right values.
        """
        # Run
        transformer = NullTransformer()

        # Assert
        assert transformer._missing_value_replacement is None
        assert transformer._missing_value_generation == 'random'
        assert transformer._min_value is None
        assert transformer._max_value is None

    def test___init__not_default(self):
        """Test the initialization passing values different than defaults.

        When arguments are passed, the attributes should be populated
        with the right values.

        Input:
            - Values different than the defaults.

        Expected Side Effects:
            - The attributes should be populated with the given values.
        """
        # Run
        transformer = NullTransformer('a_missing_value_replacement', None)

        # Assert
        assert transformer._missing_value_replacement == 'a_missing_value_replacement'
        assert transformer._missing_value_generation is None
        assert transformer._min_value is None
        assert transformer._max_value is None

    def test___init__raises_error(self):
        """Test the initialization passing invalid values for ``missing_value_generation``."""
        # Setup
        error_msg = re.escape(
            "'missing_value_generation' must be one of the following values: None, 'from_column' "
            "or 'random'."
        )

        # Run / Assert
        with pytest.raises(TransformerInputError, match=error_msg):
            NullTransformer('mean', 'None')

    def test_models_missing_values(self):
        """Test the models_missing_values method.

        Test that when ``missing_value_generation`` is ``'from_column'``, this returns
        ``True``.
        """
        # Setup
        transformer = NullTransformer('something', missing_value_generation='from_column')

        # Run
        models_missing_values = transformer.models_missing_values()

        # Assert
        assert models_missing_values is True

    def test_models_missing_values_missing_value_generation_is_none(self):
        """Test the models_missing_values method.

        Test that when ``missing_value_generation`` is other than ``'from_column'``, this returns
        ``False``.
        """
        # Setup
        none_transformer = NullTransformer('something', missing_value_generation=None)
        random_transformer = NullTransformer('something', missing_value_generation='random')

        # Run
        none_models_missing_values = none_transformer.models_missing_values()
        random_models_missing_values = random_transformer.models_missing_values()

        # Assert
        assert none_models_missing_values is False
        assert random_models_missing_values is False

    def test__get_missing_value_replacement_scalar(self):
        """Test _get_missing_value_replacement when a scalar value is passed.

        If a missing_value_replacement different from None, 'mean' or 'mode' is
        passed to __init__, that value is returned.

        Setup:
            - NullTransformer passing a specific missing_value_replacement
              that is not None, mean or mode.

        Input:
            - A Series with some values.
            - A np.array with boolean values.

        Expected Output:
            - The value passed to __init__
        """
        # Setup
        transformer = NullTransformer('a_missing_value_replacement')

        # Run
        data = pd.Series([1, np.nan, 3], name='abc')
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        assert missing_value_replacement == 'a_missing_value_replacement'

    def test__get_missing_value_replacement_none_numerical(self):
        """Test _get_missing_value_replacement when missing_value_replacement is None.

        If the missing_value_replacement is None and the data is numerical,
        the output fill value should be the mean of the input data.

        Setup:
            - NullTransformer passing with default arguments.

        Input:
            - An Series filled with integer values such that the mean
              is not contained in the series and there is at least one null.
            - A np.array of booleans indicating which values are null.

        Expected Output:
            - The mean of the inputted Series.
        """
        # Setup
        transformer = NullTransformer('mean')

        # Run
        data = pd.Series([1, 2, np.nan], name='abc')
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        assert missing_value_replacement == 1.5

    def test__get_missing_value_replacement_none_not_numerical(self):
        """Test _get_missing_value_replacement when missing_value_replacement is None.

        If the missing_value_replacement is None and the data is not numerical,
        the output fill value should be the mode of the input data.

        Setup:
            - NullTransformer with default arguments.

        Input:
            - An Series filled with string values with variable frequency and
              at least one null value.
            - A np.array of booleans indicating which values are null.

        Expected Output:
            - The most frequent value in the input series.
        """
        # Setup
        transformer = NullTransformer('mode')

        # Run
        data = pd.Series(['a', 'b', 'b', np.nan], name='abc')
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        assert missing_value_replacement == 'b'

    def test__get_missing_value_replacement_mean(self):
        """Test _get_missing_value_replacement when missing_value_replacement is mean.

        If the missing_value_replacement is mean the output fill value should be the
        mean of the input data.

        Setup:
            - NullTransformer passing 'mean' as the missing_value_replacement.

        Input:
            - A Series filled with integer values such that the mean
              is not contained in the series and there is at least one null.
            - A np.array of booleans indicating which values are null.

        Expected Output:
            - The mode of the inputted Series.
        """
        # Setup
        transformer = NullTransformer('mean')

        # Run
        data = pd.Series([1, 2, np.nan], name='abc')
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        assert missing_value_replacement == 1.5

    @patch('rdt.transformers.null.LOGGER')
    def test__get_missing_value_replacement_mean_only_nans(self, logger_mock):
        """Test when missing_value_replacement is mean and data only contains nans."""
        # Setup
        transformer = NullTransformer('mean')
        data = pd.Series([float('nan'), None, np.nan], name='abc')

        # Run
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        logger_mock.info.assert_called_once_with(
            "'missing_value_replacement' cannot be set to 'mean'"
            ' when the provided data only contains NaNs. Using 0 instead.'
        )
        assert missing_value_replacement == 0

    def test__get_missing_value_replacement_mode(self):
        """Test _get_missing_value_replacement when missing_value_replacement is 'mode'.

        If the missing_value_replacement is 'mode' the output fill value should be the
        mode of the input data.

        Setup:
            - NullTransformer passing 'mode' as the missing_value_replacement.

        Input:
            - A Series filled with integer values such that the mean
              is not contained in the series and there is at least one null.
            - A np.array of booleans indicating which values are null.

        Expected Output:
            - The most frequent value in the input series.
        """
        # Setup
        transformer = NullTransformer('mode')

        # Run
        data = pd.Series([1, 2, 2, np.nan], name='abc')
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        assert missing_value_replacement == 2

    @patch('rdt.transformers.null.LOGGER')
    def test__get_missing_value_replacement_mode_only_nans(self, logger_mock):
        """Test when missing_value_replacement is mode and data only contains nans."""
        # Setup
        transformer = NullTransformer('mode')
        data = pd.Series([float('nan'), None, np.nan], name='abc')

        # Run
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        logger_mock.info.assert_called_once_with(
            "'missing_value_replacement' cannot be set to 'mode' when "
            'the provided data only contains NaNs. Using 0 instead.'
        )
        assert missing_value_replacement == 0

    def test_fit_missing_value_generation_is_none_and_nulls(self):
        """Test fit when ``missing_value_generation`` is ``None`` and there are nulls.

        Nothing has been learned and the nulls stay as ``None``. The ``_missing_value_replacement``
        should be set to the mean.
        """
        # Setup
        transformer = NullTransformer(
            missing_value_replacement='mean',
            missing_value_generation=None
        )

        # Run
        data = pd.Series([1, 2, np.nan])
        transformer.fit(data)

        # Assert
        assert transformer.nulls is None
        assert transformer._missing_value_replacement == 1.5

    def test_fit_missing_value_generation_from_column_and_no_nulls(self):
        """Test fit when ``missing_value_generation`` is 'from_column' and there are nulls."""
        # Setup
        transformer = NullTransformer(missing_value_generation='from_column')

        # Run
        data = pd.Series(['a', 'b', 'b'])
        transformer.fit(data)

        # Assert
        assert not transformer.nulls
        assert transformer._missing_value_generation is None
        assert transformer._missing_value_replacement is None

    def test_fit_missing_value_replacement_is_random(self):
        """Test fit when ``missing_value_replacement`` is random."""
        # Setup
        transformer = NullTransformer(missing_value_replacement='random')

        # Run
        data = pd.Series([1, 2, 3])
        transformer.fit(data)

        # Assert
        assert not transformer.nulls
        assert transformer._missing_value_generation == 'random'
        assert transformer._missing_value_replacement == 'random'
        assert transformer._min_value == 1
        assert transformer._max_value == 3

    def test_fit_with_multiple_missing_value_generations(self):
        """Test fit when ``missing_value_generation`` is set to its three possibilities.

        Test that when there are multiple scenarios given, the null transformer is able to
        either learn the mode or mean to replace the value but following the
        ``missing_value_generation`` strategy.

        Notice that this test covers 5 scenarios at once.
        """
        # Setup
        missing_value_generation_random_nulls = NullTransformer(
            missing_value_replacement='mode',
            missing_value_generation='random'
        )
        missing_value_generation_random_no_nulls = NullTransformer(
            missing_value_replacement='mode',
            missing_value_generation='random'
        )
        missing_value_generation_column_nulls = NullTransformer(
            missing_value_replacement='mean',
            missing_value_generation='from_column'
        )
        missing_value_generation_column_no_nulls = NullTransformer(
            missing_value_replacement='mean',
            missing_value_generation='from_column'
        )

        missing_value_generation_none_int = NullTransformer(
            missing_value_replacement='mean',
            missing_value_generation=None
        )
        missing_value_generation_none_str = NullTransformer(
            missing_value_replacement='mode',
            missing_value_generation=None
        )

        nulls_str = pd.Series(['a', 'b', 'b', np.nan])
        no_nulls_str = pd.Series(['a', 'b', 'b', 'c'])
        nulls_int = pd.Series([1, 2, 3, np.nan])
        no_nulls_int = pd.Series([1, 2, 3, 4])

        # Run
        missing_value_generation_random_nulls.fit(nulls_str)
        missing_value_generation_random_no_nulls.fit(no_nulls_str)
        missing_value_generation_column_nulls.fit(nulls_int)
        missing_value_generation_column_no_nulls.fit(no_nulls_int)
        missing_value_generation_none_int.fit(nulls_int)
        missing_value_generation_none_str.fit(nulls_str)

        # Assert
        assert missing_value_generation_random_nulls._missing_value_generation == 'random'
        assert missing_value_generation_random_nulls.nulls
        assert missing_value_generation_random_nulls._missing_value_replacement == 'b'

        assert missing_value_generation_random_no_nulls._missing_value_generation == 'random'
        assert not missing_value_generation_random_no_nulls.nulls
        assert missing_value_generation_random_no_nulls._missing_value_replacement == 'b'

        assert missing_value_generation_column_nulls._missing_value_generation == 'from_column'
        assert missing_value_generation_column_nulls.nulls
        assert missing_value_generation_column_nulls._missing_value_replacement == 2

        assert missing_value_generation_column_no_nulls._missing_value_generation is None
        assert not missing_value_generation_column_no_nulls.nulls
        assert missing_value_generation_column_no_nulls._missing_value_replacement == 2.5

        assert missing_value_generation_none_int._missing_value_generation is None
        assert missing_value_generation_none_int.nulls is None
        assert missing_value_generation_none_int._missing_value_replacement == 2

        assert missing_value_generation_none_str._missing_value_generation is None
        assert missing_value_generation_none_str.nulls is None
        assert missing_value_generation_none_str._missing_value_replacement == 'b'

    def test_transform__missing_value_generation_from_column(self):
        """Test transform when ``_missing_value_generation`` is set to ``from_column``.

        When ``missing_value_generation`` is 'from_column', the nulls should be replaced
        by the ``_missing_value_replacement`` and another column flagging the nulls
        should be created.
        """
        # Setup
        transformer = NullTransformer(missing_value_generation='from_column')
        transformer.nulls = False
        transformer._missing_value_replacement = 'c'
        input_data = pd.Series(['a', 'b', np.nan])

        # Run
        output = transformer.transform(input_data)

        # Assert
        expected_output = np.array([
            ['a', 0.0],
            ['b', 0.0],
            ['c', 1.0],
        ], dtype=object)
        np.testing.assert_equal(expected_output, output)

    def test_transform__missing_value_generation_random(self):
        """Test transform when ``_missing_value_generation`` is set to ``random``.

        Test that when the ``_missing_value_generation`` is set to ``random``, the nulls should
        be replaced by the ``_missing_value_replacement``.
        """
        # Setup
        transformer = NullTransformer()
        transformer._missing_value_replacement = 3
        input_data = pd.Series([1, 2, np.nan])

        # Run
        output = transformer.transform(input_data)

        # Assert
        expected_output = np.array([1, 2, 3])
        np.testing.assert_equal(expected_output, output)

        modified_input_data = pd.Series([1, 2, np.nan])
        pd.testing.assert_series_equal(modified_input_data, input_data)

    @patch('rdt.transformers.null.np.random.uniform')
    def test_transform__missing_value_replacement_random(self, mock_np_random_uniform):
        """Test transform when ``_missing_value_replacement`` is set to ``random``."""
        # Setup
        transformer = NullTransformer(missing_value_replacement='random')
        input_data = pd.Series([1, 2, np.nan])
        transformer._min_value = 1
        transformer._max_value = 2
        mock_np_random_uniform.return_value = np.array([1, 1.25, 1.5])

        # Run
        output = transformer.transform(input_data)

        # Assert
        expected_output = np.array([1, 2, 1.5])
        np.testing.assert_equal(expected_output, output)

        modified_input_data = pd.Series([1, 2, np.nan])
        pd.testing.assert_series_equal(modified_input_data, input_data)
        mock_np_random_uniform.assert_called_once_with(low=1, high=2, size=3)

    def test_reverse_transform__missing_value_generation_from_column_with_nulls(self):
        """Test reverse_transform when ``missing_value_generation`` is ``from_column`` and nulls.

        When ``missing_value_generation`` is ``from_column`` and there are nulls, the second column
        in the input data should be used to decide which values to replace with nan,
        by selecting the rows where the null column value is > 0.5.
        """
        # Setup
        transformer = NullTransformer(missing_value_generation='from_column')
        transformer.nulls = True
        input_data = np.array([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.8],
        ])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, 0.4, np.nan, np.nan])
        pd.testing.assert_series_equal(expected_output, output)

    def test_reverse_transform__missing_value_generation_from_column_no_nulls(self):
        """Test reverse_transform when ``missing_value_generation`` is ``from_column``, no nulls.

        When ``missing_value_generation`` is ``from_column`` but no nulls are found, the second
        column of the input data must be dropped and the first one returned as a ``pd.Series``
        without having been modified.
        """
        # Setup
        transformer = NullTransformer(missing_value_generation='from_column')
        transformer.nulls = False
        input_data = np.array([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.8],
        ])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8])
        pd.testing.assert_series_equal(expected_output, output)

    @patch('rdt.transformers.null.np.random')
    def test_reverse_transform__missing_value_generation_random_with_nulls(self, random_mock):
        """Test reverse_transform when ``missing_value_generation`` is ``random`` and nulls.

        When ``missing_value_generation`` is ``random`` and there are nulls, a ``_null_percentage``
        of values should randomly be replaced with ``np.nan``.
        """
        # Setup
        transformer = NullTransformer(missing_value_generation='random')
        transformer.nulls = True
        transformer._null_percentage = 0.5
        input_data = np.array([0.0, 0.2, 0.4, 0.6])
        random_mock.random.return_value = np.array([1, 1, 0, 1])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, np.nan, 0.6])
        pd.testing.assert_series_equal(expected_output, output)

    def test_reverse_transform__missing_value_generation_random_no_nulls(self):
        """Test reverse_transform when missing_value_generation is ``random`` and no nulls.

        When ``missing_value_generation`` is ``random`` and there are no nulls, the input data
        is not modified at all.
        """
        # Setup
        transformer = NullTransformer(missing_value_generation='random')
        transformer.nulls = False
        transformer._missing_value_replacement = 0.4
        input_data = np.array([0.0, 0.2, 0.4, 0.6])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, 0.4, 0.6])
        pd.testing.assert_series_equal(expected_output, output)
