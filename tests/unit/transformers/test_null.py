"""Unit tests for the NullTransformer."""

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import NullTransformer


class TestNullTransformer:

    def test___init__default(self):
        """Test the initialization without passing any default arguments.

        When no arguments are passed, the attributes should be populated
        with the right values.

        Input:
            - nothing

        Expected Side Effects:
            - The `fill_value` attribute should be `None`.
            - The `null_column` attribute should be `None`.
            - The `copy` attribute should be `False`.
        """
        # Run
        transformer = NullTransformer()

        # Assert
        assert transformer.fill_value is None
        assert transformer.null_column is None
        assert not transformer.copy

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
        transformer = NullTransformer('a_fill_value', False, True)

        # Assert
        assert transformer.fill_value == 'a_fill_value'
        assert not transformer.null_column
        assert transformer.copy

    def test_creates_null_column(self):
        """Test the creates_null_column method.

        If the `null_column` attributes evalutes to True, the `create_null_column`
        method should return the same value.

        Setup:
            - Create an instance and set _null_column to True

        Expected Output:
            - True
        """
        # Setup
        transformer = NullTransformer('something', null_column=True)
        transformer._null_column = True

        # Run
        creates_null_column = transformer.creates_null_column()

        # Assert
        assert creates_null_column

    def test__get_fill_value_scalar(self):
        """Test _get_fill_value when a scalar value is passed.

        If a fill_value different from None, 'mean' or 'mode' is
        passed to __init__, that value is returned.

        Setup:
            - NullTransformer passing a specific fill_value
              that is not None, mean or mode.

        Input:
            - A Series with some values.
            - A np.array with boolean values.

        Expected Output:
            - The value passed to __init__
        """
        # Setup
        transformer = NullTransformer('a_fill_value')

        # Run
        data = pd.Series([1, np.nan, 3], name='abc')
        null_values = np.array([False, True, False])
        fill_value = transformer._get_fill_value(data, null_values)

        # Assert
        assert fill_value == 'a_fill_value'

    def test__get_fill_value_all_nulls(self):
        """Test _get_fill_value when all the values are null.

        If the fill_value is not a scalar value and all the data
        values are null, the output should be 0.

        Setup:
            - NullTransformer passing 'mean' as the fill_value.

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
        null_values = np.array([True, True, True])
        fill_value = transformer._get_fill_value(data, null_values)

        # Assert
        assert fill_value == 0

    def test__get_fill_value_none_numerical(self):
        """Test _get_fill_value when fill_value is None and data is numerical.

        If the fill_value is None and the data is numerical,
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
        transformer = NullTransformer()

        # Run
        data = pd.Series([1, 2, np.nan], name='abc')
        null_values = np.array([False, False, True])
        fill_value = transformer._get_fill_value(data, null_values)

        # Assert
        assert fill_value == 1.5

    def test__get_fill_value_none_not_numerical(self):
        """Test _get_fill_value when fill_value is None and data is not numerical.

        If the fill_value is None and the data is not numerical,
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
        transformer = NullTransformer()

        # Run
        data = pd.Series(['a', 'b', 'b', np.nan], name='abc')
        null_values = np.array([False, False, False, True])
        fill_value = transformer._get_fill_value(data, null_values)

        # Assert
        assert fill_value == 'b'

    def test__get_fill_value_mean(self):
        """Test _get_fill_value when fill_value is mean.

        If the fill_value is mean the output fill value should be the
        mean of the input data.

        Setup:
            - NullTransformer passing 'mean' as the fill_value.

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
        null_values = np.array([False, False, True])
        fill_value = transformer._get_fill_value(data, null_values)

        # Assert
        assert fill_value == 1.5

    def test__get_fill_value_mode(self):
        """Test _get_fill_value when fill_value is 'mode'.

        If the fill_value is 'mode' the output fill value should be the
        mode of the input data.

        Setup:
            - NullTransformer passing 'mode' as the fill_value.

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
        null_values = np.array([False, False, False, True])
        fill_value = transformer._get_fill_value(data, null_values)

        # Assert
        assert fill_value == 2

    def test_fit_null_column_none_and_nulls(self):
        """Test fit when null column is none and there are nulls.

        If there are nulls in the data and null_column was given as None,
        then the _null_column attribute should be set to True.
        Also validate that the null attribute and the _fill_value attributes
        are set accordingly.

        Setup:
            - A NullTransformer with default arguments.

        Input:
            - pd.Series of integers that contains nulls.

        Expected Side Effects:
            - the null_column attribute should be set to True.
            - the nulls attribute should be set to True.
            - the fill_value should be set to the mean of the given integers.
        """
        # Setup
        transformer = NullTransformer()

        # Run
        data = pd.Series([1, 2, np.nan])
        transformer.fit(data)

        # Assert
        assert transformer.nulls
        assert transformer._null_column
        assert transformer._fill_value == 1.5

    def test_fit_null_column_none_and_no_nulls(self):
        """Test fit when null column is none and there are NO nulls.

        If there are no nulls in the data and null_column was given as None,
        then the _null_column attribute should be set to False.
        Also validate that the null attribute and the _fill_value attributes
        are set accordingly.

        Setup:
            - A NullTransformer with default arguments.

        Input:
            - pd.Series of strings that contains no nulls.

        Expected Side Effects:
            - the null_column attribute should be set to False.
            - the nulls attribute should be set to False.
            - the fill_value should be set to the mode of the given strings.
        """
        # Setup
        transformer = NullTransformer()

        # Run
        data = pd.Series(['a', 'b', 'b'])
        transformer.fit(data)

        # Assert
        assert not transformer.nulls
        assert not transformer._null_column
        assert transformer._fill_value == 'b'

    def test_fit_null_column_not_none(self):
        """Test fit when null column is set to True/False.

        If null_column is set to True or False, the _null_column should
        get that value regardless of whether there are nulls or not.

        Notice that this test covers 4 scenarios at once.

        Setup:
            - 4 NullTransformer intances, 2 of them passing False for the null_column
              and 2 of them passing True.

        Input:
            - 2 pd.Series, one containing nulls and the other not containing nulls.

        Expected Side Effects:
            - the _null_column attribute should be set to True or False as indicated
              in the Transformer creation.
            - the nulls attribute should be True or False depending on whether
              the input data contains nulls or not.
        """
        # Setup
        null_column_false_nulls = NullTransformer(null_column=False)
        null_column_false_no_nulls = NullTransformer(null_column=False)
        null_column_true_nulls = NullTransformer(null_column=True)
        null_column_true_no_nulls = NullTransformer(null_column=True)
        nulls_str = pd.Series(['a', 'b', 'b', np.nan])
        no_nulls_str = pd.Series(['a', 'b', 'b', 'c'])
        nulls_int = pd.Series([1, 2, 3, np.nan])
        no_nulls_int = pd.Series([1, 2, 3, 4])

        # Run
        null_column_false_nulls.fit(nulls_str)
        null_column_false_no_nulls.fit(no_nulls_str)
        null_column_true_nulls.fit(nulls_int)
        null_column_true_no_nulls.fit(no_nulls_int)

        # Assert
        assert not null_column_false_nulls._null_column
        assert null_column_false_nulls.nulls
        assert null_column_false_nulls._fill_value == 'b'

        assert not null_column_false_no_nulls._null_column
        assert not null_column_false_no_nulls.nulls
        assert null_column_false_no_nulls._fill_value == 'b'

        assert null_column_true_nulls._null_column
        assert null_column_true_nulls.nulls
        assert null_column_true_nulls._fill_value == 2

        assert null_column_true_no_nulls._null_column
        assert not null_column_true_no_nulls.nulls
        assert null_column_true_no_nulls._fill_value == 2.5

    def test_transform__null_column_false_copy_false(self):
        """Test transform when _null_column is False and copy is False.

        When the _null_column is false, the nulls should be replaced
        by the _fill_value.
        When copy is False, the original data is affected.

        Setup:
            - NullTransformer instance with _null_column set to False,
              _fill_value set to a scalar value, and copy set to False.

        Input:
            - A pd.Series of integers with nulls.

        Expected Output:
            - Same data as the input, replacing the nulls with the
              scalar value.

        Expected Side Effects:
            - The input data has the null values replaced.
        """
        # Setup
        transformer = NullTransformer()
        transformer._null_column = False
        transformer._fill_value = 3
        transformer.copy = False
        input_data = pd.Series([1, 2, np.nan])

        # Run
        output = transformer.transform(input_data)

        # Assert
        expected_output = np.array([1, 2, 3])
        np.testing.assert_equal(expected_output, output)

        modified_input_data = pd.Series([1, 2, 3], dtype=np.float64)
        pd.testing.assert_series_equal(modified_input_data, input_data)

    def test_transform__null_column_true_copy_false(self):
        """Test transform when _null_column and copy is False.

        When _null_column, the nulls should be replaced
        by the _fill_value and another column flagging the nulls
        should be created.
        When copy is False, the original data is affected.

        Setup:
            - NullTransformer instance with _null_column set to True,
              _fill_value set to a scalar value, and copy set to False.

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
        transformer._null_column = True
        transformer._fill_value = 'c'
        transformer.copy = False
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

        modified_input_data = pd.Series(['a', 'b', 'c'])
        pd.testing.assert_series_equal(modified_input_data, input_data)

    def test_transform__null_column_false_copy_true(self):
        """Test transform when _null_column is False and copy.

        When the _null_column is false, the nulls should be replaced
        by the _fill_value.
        When copy, the original data is not affected.

        Setup:
            - NullTransformer instance with _null_column set to False,
              _fill_value set to a scalar value, and copy set to True.

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
        transformer._null_column = False
        transformer._fill_value = 3
        transformer.copy = True
        input_data = pd.Series([1, 2, np.nan])

        # Run
        output = transformer.transform(input_data)

        # Assert
        expected_output = np.array([1, 2, 3])
        np.testing.assert_equal(expected_output, output)

        modified_input_data = pd.Series([1, 2, np.nan])
        pd.testing.assert_series_equal(modified_input_data, input_data)

    def test_transform__null_column_true_copy_true(self):
        """Test transform when _null_column and copy.

        When _null_column, the nulls should be replaced
        by the _fill_value and another column flagging the nulls
        should be created.
        When copy, the original data is not affected.

        Setup:
            - NullTransformer instance with _null_column set to True,
              _fill_value set to a scalar value, and copy set to True.

        Input:
            - A pd.Series of strings with nulls.

        Expected Output:
            - Exactly the same as the input, replacing the nulls with the
              scalar value.

        Expected Side Effects:
            - The input data has not been modified.
        """
        # Setup
        transformer = NullTransformer()
        transformer.nulls = False
        transformer._null_column = True
        transformer._fill_value = 'c'
        transformer.copy = True
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

        modified_input_data = pd.Series(['a', 'b', np.nan])
        pd.testing.assert_series_equal(modified_input_data, input_data)

    def test_transform__irreversible_warning(self):
        """Test transform when _null_column is False and fill_value is in data.

        When _null_column is False, and the fill_value exists in the data
        before transforming, a warning is raised.

        Setup:
            - NullTransformer instance with _null_column set to False
              and _fill_value set to a scalar value.

        Input:
            - A pd.Series of strings with nulls, containing the fill_value.

        Expected Output:
            - Exactly the same as the input, replacing the nulls with the
              scalar value.

        Expected Side Effects:
            - A UserWarning is raised.
        """
        # Setup
        transformer = NullTransformer()
        transformer._null_column = False
        transformer._fill_value = 'b'
        input_data = pd.Series(['a', 'b', np.nan])

        # Run
        with pytest.warns(UserWarning):
            output = transformer.transform(input_data)

        # Assert
        expected_output = np.array(['a', 'b', 'b'])
        np.testing.assert_equal(expected_output, output)

    def test_reverse_transform__null_column_true_nulls_true_copy_true(self):
        """Test reverse_transform when _null_column, nulls and copy are True.

        When _null_column and nulls attributes are both True, the second column
        in the input data should be used to decide which values to replace
        with nan, by selecting the rows where the null column value is > 0.5.
        If copy, the input data should not be modified.

        Setup:
            - NullTransformer instance with _null_column, nulls and copy
              attributes set to True.

        Input:
            - 2d numpy array with variate float values.

        Expected Output:
            - pd.Series containing the first column from the input data
              with the values indicated by the first column replaced by nans.

        Expected Side Effects:
            - the input data should not have been modified.
        """
        # Setup
        transformer = NullTransformer()
        transformer._null_column = True
        transformer.nulls = True
        transformer.copy = True
        input_data = np.array([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.8],
        ])
        input_data_copy = input_data.copy()

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, 0.4, np.nan, np.nan])
        pd.testing.assert_series_equal(expected_output, output)

        np.testing.assert_equal(input_data, input_data_copy)

    def test_reverse_transform__null_column_true_nulls_true_copy_false(self):
        """Test reverse_transform when _null_column and nulls are True, and copy is False.

        When _null_column and nulls attributes are both True, the second column
        in the input data should be used to decide which values to replace
        with nan, by selecting the rows where the null column value is > 0.5.
        If copy is False, the input data should be modified in place.

        Setup:
            - NullTransformer instance with _null_column and nulls
              attributes set to True, and copy set to False.

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
        transformer._null_column = True
        transformer.nulls = True
        transformer.copy = False
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

        modified_input_data = np.array([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.4, 0.4],
            [np.nan, 0.6],
            [np.nan, 0.8],
        ])
        np.testing.assert_equal(modified_input_data, input_data)

    def test_reverse_transform__null_column_true_nulls_false(self):
        """Test reverse_transform when _null_column and nulls is False.

        When _null_column but nulls are False, the second column of the
        input data must be dropped and the first one returned as a Series without
        having been modified.

        Setup:
            - NullTransformer instance with _null_column set to True and nulls
              attribute set to False

        Input:
            - 2d numpy array with variate float values.

        Expected Output:
            - pd.Series containing the first column from the input data unmodified.
        """
        # Setup
        transformer = NullTransformer()
        transformer._null_column = True
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

    def test_reverse_transform__null_column_false_nulls_true(self):
        """Test reverse_transform when _null_column is False and nulls.

        When _null_column is False and the nulls attribute, the
        values in the input data that match the _fill_value attribute are
        replaced by nans.

        Setup:
            - NullTransformer instance with _null_column set to False and nulls
              attribute set to True, and with the _fill_value set to a scalar.

        Input:
            - 1d numpy array with variate float values, containing the _fill_value
              among them.

        Expected Output:
            - pd.Series containing the same data as input, with the values that
              match the _fill_value replaced by nans.
        """
        # Setup
        transformer = NullTransformer()
        transformer._null_column = False
        transformer.nulls = True
        transformer._fill_value = 0.4
        input_data = np.array([0.0, 0.2, 0.4, 0.6])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, np.nan, 0.6])
        pd.testing.assert_series_equal(expected_output, output)

    def test_reverse_transform__null_column_false_nulls_false(self):
        """Test reverse_transform when _null_column is False and nulls.

        When _null_column is False and the nulls attribute is also False, the
        input data is not modified at all.

        Setup:
            - NullTransformer instance with _null_column and nulls attributes
              set to False, and the _fill_value set to a scalar value.

        Input:
            - 1d numpy array with variate float values, containing the _fill_value
              among them.

        Expected Output:
            - pd.Series containing the same data as input, without modification.
        """
        # Setup
        transformer = NullTransformer()
        transformer._null_column = False
        transformer.nulls = False
        transformer._fill_value = 0.4
        input_data = np.array([0.0, 0.2, 0.4, 0.6])

        # Run
        output = transformer.reverse_transform(input_data)

        # Assert
        expected_output = pd.Series([0.0, 0.2, 0.4, 0.6])
        pd.testing.assert_series_equal(expected_output, output)
