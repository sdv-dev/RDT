"""Unit tests for the NullTransformer."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from rdt.transformers import NullTransformer


class TestNullTransformer:

    def test___init__default(self):
        """Test the initialization without passing any default arguments.

        When no arguments are passed, the attributes should be populated
        with the right values.

        Input:
            - nothing

        Expected Side Effects:
            - The `_missing_value_replacement` attribute should be `None`.
            - The `_model_missing_values` attribute should be `False`.
        """
        # Run
        transformer = NullTransformer()

        # Assert
        assert transformer._missing_value_replacement is None
        assert not transformer._model_missing_values

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
        transformer = NullTransformer('a_missing_value_replacement', False)

        # Assert
        assert transformer._missing_value_replacement == 'a_missing_value_replacement'
        assert not transformer._model_missing_values

    def test_models_missing_values(self):
        """Test the models_missing_values method.

        If the `model_missing_values` attributes evalutes to True, the
        `create_model_missing_values` method should return the same value.

        Setup:
            - Create an instance and set _model_missing_values to True

        Expected Output:
            - True
        """
        # Setup
        transformer = NullTransformer('something', model_missing_values=True)
        transformer._model_missing_values = True

        # Run
        models_missing_values = transformer.models_missing_values()

        # Assert
        assert models_missing_values

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

    def test__get_missing_value_replacement_all_nulls(self):
        """Test _get_missing_value_replacement when all the values are null.

        If the missing_value_replacement is not a scalar value and all the data
        values are null, the output be the mean, which is `np.nan`.

        Setup:
            - NullTransformer passing 'mean' as the missing_value_replacement.

        Input:
            - A Series filled with nan values.
            - A np.array of all True values.

        Expected Output:
            - 0
        """
        # Setup
        transformer = NullTransformer('mean')

        # Run
        data = pd.Series([np.nan, np.nan, np.nan], name='abc')
        missing_value_replacement = transformer._get_missing_value_replacement(data)

        # Assert
        assert missing_value_replacement is np.nan

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

    def test_fit_model_missing_values_none_and_nulls(self):
        """Test fit when null column is none and there are nulls.

        If there are nulls in the data and model_missing_values was given as None,
        then the _model_missing_values attribute should be set to True.
        Also validate that the null attribute and the _missing_value_replacement attributes
        are set accordingly.

        Setup:
            - A NullTransformer with default arguments.

        Input:
            - pd.Series of integers that contains nulls.

        Expected Side Effects:
            - the model_missing_values attribute should be set to True.
            - the nulls attribute should be set to True.
            - the missing_value_replacement should be set to the mean of the given integers.
        """
        # Setup
        transformer = NullTransformer(missing_value_replacement='mean', model_missing_values=True)

        # Run
        data = pd.Series([1, 2, np.nan])
        transformer.fit(data)

        # Assert
        assert transformer.nulls
        assert transformer._model_missing_values
        assert transformer._missing_value_replacement == 1.5

    def test_fit_model_missing_values_none_and_no_nulls(self):
        """Test fit when null column is none and there are NO nulls.

        If there are no nulls in the data and model_missing_values was given as ``False``,
        then the _model_missing_values attribute should be set to ``False``.
        Also validate that the null attribute and the ``_missing_value_replacement`` attributes
        are set accordingly.

        Setup:
            - A NullTransformer with default arguments.

        Input:
            - pd.Series of strings that contains no nulls.

        Expected Side Effects:
            - the model_missing_values attribute should be set to False.
            - the nulls attribute should be set to False.
            - the missing_value_replacement should be set to ``np.nan``, default.
        """
        # Setup
        transformer = NullTransformer()

        # Run
        data = pd.Series(['a', 'b', 'b'])
        transformer.fit(data)

        # Assert
        assert not transformer.nulls
        assert not transformer._model_missing_values
        assert transformer._missing_value_replacement is None

    def test_fit_model_missing_values_not_none(self):
        """Test fit when null column is set to True/False.

        If model_missing_values is set to True or False, the _model_missing_values should
        get that value regardless of whether there are nulls or not.

        Notice that this test covers 4 scenarios at once.

        Setup:
            - 4 NullTransformer intances, 2 of them passing False for the model_missing_values
              and 2 of them passing True.

        Input:
            - 2 pd.Series, one containing nulls and the other not containing nulls.

        Expected Side Effects:
            - the _model_missing_values attribute should be set to True or False as indicated
              in the Transformer creation.
            - the nulls attribute should be True or False depending on whether
              the input data contains nulls or not.
        """
        # Setup
        model_missing_values_false_nulls = NullTransformer(
            missing_value_replacement='mode',
            model_missing_values=False
        )
        model_missing_values_false_no_nulls = NullTransformer(
            missing_value_replacement='mode',
            model_missing_values=False
        )
        model_missing_values_true_nulls = NullTransformer(
            missing_value_replacement='mean',
            model_missing_values=True
        )
        model_missing_values_true_no_nulls = NullTransformer(
            missing_value_replacement='mean',
            model_missing_values=True
        )
        nulls_str = pd.Series(['a', 'b', 'b', np.nan])
        no_nulls_str = pd.Series(['a', 'b', 'b', 'c'])
        nulls_int = pd.Series([1, 2, 3, np.nan])
        no_nulls_int = pd.Series([1, 2, 3, 4])

        # Run
        model_missing_values_false_nulls.fit(nulls_str)
        model_missing_values_false_no_nulls.fit(no_nulls_str)
        model_missing_values_true_nulls.fit(nulls_int)
        model_missing_values_true_no_nulls.fit(no_nulls_int)

        # Assert
        assert not model_missing_values_false_nulls._model_missing_values
        assert model_missing_values_false_nulls.nulls
        assert model_missing_values_false_nulls._missing_value_replacement == 'b'

        assert not model_missing_values_false_no_nulls._model_missing_values
        assert not model_missing_values_false_no_nulls.nulls
        assert model_missing_values_false_no_nulls._missing_value_replacement == 'b'

        assert model_missing_values_true_nulls._model_missing_values
        assert model_missing_values_true_nulls.nulls
        assert model_missing_values_true_nulls._missing_value_replacement == 2

        assert not model_missing_values_true_no_nulls._model_missing_values
        assert not model_missing_values_true_no_nulls.nulls
        assert model_missing_values_true_no_nulls._missing_value_replacement == 2.5

    def test_transform__model_missing_values_true(self):
        """Test transform when _model_missing_values.

        When _model_missing_values, the nulls should be replaced
        by the _missing_value_replacement and another column flagging the nulls
        should be created.

        Setup:
            - NullTransformer instance with _model_missing_values set to True,
              _missing_value_replacement set to a scalar value.

        Input:
            - A pd.Series of strings with nulls.

        Expected Output:
            - Exactly the same as the input, replacing the nulls with the
              scalar value.

        Expected Side Effects:
            - The input data has the null values replaced.
        """
        # Setup
        transformer = NullTransformer()
        transformer.nulls = False
        transformer._model_missing_values = True
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

    def test_transform__model_missing_values_false(self):
        """Test transform when _model_missing_values is False.

        When the _model_missing_values is false, the nulls should be replaced
        by the _missing_value_replacement.

        Setup:
            - NullTransformer instance with _model_missing_values set to False,
              _missing_value_replacement set to a scalar value.

        Input:
            - A pd.Series of integers with nulls.

        Expected Output:
            - Same data as the input, replacing the nulls with the
              scalar value.

        Expected Side Effects:
            - The input data has not been modified.
        """
        # Setup
        transformer = NullTransformer()
        transformer._model_missing_values = False
        transformer._missing_value_replacement = 3
        input_data = pd.Series([1, 2, np.nan])

        # Run
        output = transformer.transform(input_data)

        # Assert
        expected_output = np.array([1, 2, 3])
        np.testing.assert_equal(expected_output, output)

        modified_input_data = pd.Series([1, 2, np.nan])
        pd.testing.assert_series_equal(modified_input_data, input_data)

    def test_reverse_transform__model_missing_values_true_nulls_true(self):
        """Test reverse_transform when _model_missing_values and nulls are True.

        When _model_missing_values and nulls attributes are both True, the second column
        in the input data should be used to decide which values to replace
        with nan, by selecting the rows where the null column value is > 0.5.

        Setup:
            - NullTransformer instance with _model_missing_values and nulls
              attributes set to True.

        Input:
            - 2d numpy array with variate float values.

        Expected Output:
            - pd.Series containing the first column from the input data
              with the values indicated by the first column replaced by nans.

        Expected Side Effects:
            - the input data should have been modified.
        """
        # Setup
        transformer = NullTransformer()
        transformer._model_missing_values = True
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

    def test_reverse_transform__model_missing_values_true_nulls_false(self):
        """Test reverse_transform when _model_missing_values and nulls is False.

        When _model_missing_values but nulls are False, the second column of the
        input data must be dropped and the first one returned as a Series without
        having been modified.

        Setup:
            - NullTransformer instance with _model_missing_values set to True and nulls
              attribute set to False

        Input:
            - 2d numpy array with variate float values.

        Expected Output:
            - pd.Series containing the first column from the input data unmodified.
        """
        # Setup
        transformer = NullTransformer()
        transformer._model_missing_values = True
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
    def test_reverse_transform__model_missing_values_false_nulls_true(self, random_mock):
        """Test reverse_transform when _model_missing_values is False and nulls.

        When _model_missing_values is False and the nulls attribute, a ``_null_percentage``
        of values should randomly be replaced with ``np.nan``.

        Setup:
            - NullTransformer instance with _model_missing_values set to False and nulls
              attribute set to True.
            - A mock for ``np.random``.

        Input:
            - 1d numpy array with variate float values.

        Expected Output:
            - pd.Series containing the same data as input, with the random values
            replaced with ``np.nan``.
        """
        # Setup
        transformer = NullTransformer()
        transformer._model_missing_values = False
        transformer.nulls = True
        transformer._null_percentage = 0.5
        input_data = np.array([0.0, 0.2, 0.4, 0.6])
        random_mock.random.return_value = np.array([1, 1, 0, 1])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, np.nan, 0.6])
        pd.testing.assert_series_equal(expected_output, output)

    def test_reverse_transform__model_missing_values_false_nulls_false(self):
        """Test reverse_transform when _model_missing_values is False and nulls.

        When _model_missing_values is False and the nulls attribute is also False, the
        input data is not modified at all.

        Setup:
            - NullTransformer instance with _model_missing_values and nulls attributes
              set to False, and the _missing_value_replacement set to a scalar value.

        Input:
            - 1d numpy array with variate float values, containing the _missing_value_replacement
              among them.

        Expected Output:
            - pd.Series containing the same data as input, without modification.
        """
        # Setup
        transformer = NullTransformer()
        transformer._model_missing_values = False
        transformer.nulls = False
        transformer._missing_value_replacement = 0.4
        input_data = np.array([0.0, 0.2, 0.4, 0.6])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, 0.4, 0.6])
        pd.testing.assert_series_equal(expected_output, output)
