from unittest.mock import patch

import pandas as pd
import pytest

from rdt.transformers.base import BaseTransformer


class TestBaseTransformer:

    def test_get_subclasses(self):
        """Test the ``get_subclasses`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        pass  # TODO: not sure how to test this one.

    def test_get_input_type(self):
        """Test the ``get_input_type`` method.

        Setup:
            - mock the ``BaseTransformer``'s ``INPUT_TYPE`` attribute.

        Output:
            - the string stored in the ``INPUT_TYPE`` attribute.
        """
        with patch.object(BaseTransformer, 'INPUT_TYPE', 'categorical'):
            # Run
            input_type = BaseTransformer.get_input_type()

            # Assert
            expected = 'categorical'
            assert input_type == expected

    def test__add_prefix_none(self):
        """Test the ``_add_prefix`` method when passed a ``None``.

        The method should return ``None`` when ``None`` is passed as an argument.

        Input:
            - a None value.

        Output:
            - a None value.
        """
        # Setup
        dictionary = None
        base_transformer = BaseTransformer()

        # Run
        output = base_transformer._add_prefix(dictionary)

        # Assert
        expected = None
        assert output == expected

    def test__add_prefix_dictionary(self):
        """Test the ``_add_prefix`` method when passed a dictionary.

        Setup:
            - set the ``column_prefix`` of the ``BaseTransformer`` to ``'colulumn_name'``.

        Input:
            - a dictionary of strings to strings.

        Output:
            - the input dictionary with ``column_prefix`` added to the beginning of the keys.
        """
        # Setup
        dictionary = {
            'day': 'numerical',
            'month': 'categorical',
            'year': 'numerical'
        }
        transformer = BaseTransformer()
        transformer.column_prefix = 'column_name'

        # Run
        output = transformer._add_prefix(dictionary)

        # Assert
        expected = {
            'column_name.day': 'numerical',
            'column_name.month': 'categorical',
            'column_name.year': 'numerical'
        }
        assert output == expected

    def test_get_output_types(self):
        """Test the ``get_output_types`` method.

        Setup:
            - set the ``column_prefix`` of the ``BaseTransformer`` to ``'colulumn_name'``.
            - set the ``OUTPUT_TYPES`` of the ``BaseTransformer`` to a dictionary.

        Output:
            - the dictionary set in ``OUTPUT_TYPES`` with the ``column_prefix`` string
            added to the beginning of the keys.
        """
        # Setup
        base_transformer = BaseTransformer()
        base_transformer.column_prefix = 'column_name'
        base_transformer.OUTPUT_TYPES = {
            'value': 'numerical'
        }

        # Run
        output = base_transformer.get_output_types()

        # Assert
        expected = {
            'column_name.value': 'numerical'
        }
        assert output == expected

    def test_is_transform_deterministic(self):
        """Test the ``is_transform_deterministic`` method.

        Setup:
            - set the ``DETERMINISTIC_TRANSFORM`` of the ``BaseTransformer`` to True.

        Output:
            - the booloan value stored in ``DETERMINISTIC_TRANSFORM``.
        """
        # Setup
        base_transformer = BaseTransformer()
        base_transformer.DETERMINISTIC_TRANSFORM = True

        # Run
        output = base_transformer.is_transform_deterministic()

        # Assert
        expected = True
        assert output == expected

    def test_is_reverse_deterministic(self):
        """Test the ``is_reverse_deterministic`` method.

        Setup:
            - set the ``DETERMINISTIC_REVERSE`` of the ``BaseTransformer`` to True.

        Output:
            - the booloan value stored in ``DETERMINISTIC_REVERSE``.
        """
        # Setup
        base_transformer = BaseTransformer()
        base_transformer.DETERMINISTIC_REVERSE = True

        # Run
        output = base_transformer.is_reverse_deterministic()

        # Assert
        expected = True
        assert output == expected

    def test_is_composition_identity(self):
        """Test the ``is_composition_identity`` method.

        Setup:
            - set the ``COMPOSITION_IS_IDENTITY`` of the ``BaseTransformer`` to True.

        Output:
            - the booloan value stored in ``COMPOSITION_IS_IDENTITY``.
        """
        # Setup
        base_transformer = BaseTransformer()
        base_transformer.COMPOSITION_IS_IDENTITY = True

        # Run
        output = base_transformer.is_composition_identity()

        # Assert
        expected = True
        assert output == expected

    def test_get_next_transformers(self):
        """Test the ``get_next_transformers`` method.

        Setup:
            - set the ``column_prefix`` of the ``BaseTransformer`` to a string.
            - set the ``NEXT_TRANSFORMERS`` of the ``BaseTransformer`` to a dictionary.

        Output:
            - the dictionary set in ``NEXT_TRANSFORMERS`` with the ``column_prefix`` string
            added to the beginning of the keys.
        """
        # Setup
        base_transformer = BaseTransformer()
        base_transformer.column_prefix = 'column_name'
        base_transformer.NEXT_TRANSFORMERS = {
            'value': 'NullTransformer'
        }

        # Run
        output = base_transformer.get_next_transformers()

        # Assert
        expected = {
            'column_name.value': 'NullTransformer'
        }
        assert output == expected

    def test__store_columns_list(self):
        """Test the ``_store_columns`` method when passed a list.

        The method should store the passed columns in the ``columns`` attribute.

        Input:
            - a data frame.
            - a list of a subset of the columns of the dataframe.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['a', 'b']
        base_transformer = BaseTransformer()

        # Run
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns

        # Assert
        expected = ['a', 'b']
        assert stored_columns == expected

    def test__store_columns_tuple(self):
        """Test the ``_store_columns`` method when passed a tuple.

        The method should first convert the tuple to list, an then store the passed columns
        in the ``columns`` attribute.

        Input:
            - a data frame.
            - a tuple of a subset of the columns of the dataframe.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ('a', 'b')
        base_transformer = BaseTransformer()

        # Run
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns

        # Assert
        expected = ['a', 'b']
        assert stored_columns == expected

    def test__store_columns_string(self):
        """Test the ``_store_columns`` method when passed a string.

        The method should first convert the passed string into a one element list, and then
        store this one column in the ``columns`` attribute.

        Input:
            - a data frame.
            - a string with the name of one of the columns of the dataframe.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = 'a'
        base_transformer = BaseTransformer()

        # Run
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns

        # Assert
        expected = ['a']
        assert stored_columns == expected

    def test__store_columns_missing(self):
        """Test the ``_store_columns`` method when passed a missing column.

        When the passed column does not exist in the dataframe, it should raise a ``KeyError``.

        Input:
            - a data frame.
            - a list of column names, where at least one of the columns is not
            present in the dataframe.

        Raises:
            - ``KeyError``, with the appropriate error message.
        """
        # Setup
        data = pd.DataFrame()
        columns = ['a', 'b']
        base_transformer = BaseTransformer()
        missing = set(columns)
        error_msg = (f'Columns {missing} were not present in the data.')

        # Run / Assert
        with pytest.raises(KeyError, match=error_msg):
            base_transformer._store_columns(columns, data)

    def test__get_columns_data_multiple_columns(self):
        """Test the ``_get_columns_data`` method.

        The method should select the passed columns from the passed data.

        Input:
            - a dataframe.
            - a list of a subset of the columns of the dataframe.

        Output:
            - the passed dataframe, but containing only the passed columns.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['a', 'b']

        # Run
        columns_data = BaseTransformer._get_columns_data(data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(columns_data, expected)

    def test__get_columns_data_single_column(self):
        """Test the ``_get_columns_data`` method when passed a sigle column.

        The method should select the passed column from the passed data, and convert it
        into a pandas series.

        Input:
            - a dataframe.
            - a list of one column from the dataframe.

        Output:
            - a pandas series, corresponding to the passed column from the dataframe.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['b']

        # Run
        columns_data = BaseTransformer._get_columns_data(data, columns)

        # Assert
        expected = pd.Series([4, 5, 6], name='b')
        pd.testing.assert_series_equal(columns_data, expected)

    def test__set_columns_data_preserves_order_series(self):
        """Test the ``_set_columns_data`` method.

        The method should not reorder the rows from the ``columns_data``
        parameter if it is a ``Series`` and the ``data`` has a different index.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a Series with a sequential index.
            - columns will have the column name of the Series.

        Expected behavior:
            - Data should have the values from columns_data in the same order
            as they were in columns_data.
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['c']
        columns_data = pd.Series([7, 8, 9], name='c')

        BaseTransformer._set_columns_data(data, columns_data, columns)

        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__set_columns_data_preserves_order_dataframe(self):
        """Test the ``_set_columns_data`` method.

        The method should not reorder the rows from the ``columns_data``
        parameter if it is a ``DataFrame`` and the ``data`` has a different index.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a Series with a sequential index.
            - columns will have the column name of the Series.

        Expected behavior:
            - Data should have the values from columns_data in the same order
            as they were in columns_data.
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['c', 'd']
        columns_data = pd.DataFrame({
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        })

        BaseTransformer._set_columns_data(data, columns_data, columns)

        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__build_output_columns(self):
        """Test the ``_build_output_columns`` method.

        Setup:
            - set the ``columns`` attribute to a list of a subset of the columns in the data.

        Input:
            - a dataframe.

        Expected behavior:
            - ``column_prefix`` should be set to the elements stored in ``columns`` joined by
            hashtags (e.g. ['a', 'b'] -> 'a#b').
            - ``output_columns`` should be set to a list of the keys of what's returned from the
            ``get_output_types`` method.
        """

    def test__build_output_columns_generated_already_exist(self):
        """Test the ``_build_output_columns`` method.

        When this method generates column names that already exist in the data, it should
        keep adding hashtags at the end of the column name, until the column name becomes unique.

        Setup:
            - set the ``columns`` attribute to a list of a subset of the columns in the data.

        Input:
            - a dataframe where the generated column name already exist (e.g. ['a', 'b', 'a#b']).

        Expected behavior:
            - ``column_prefix`` should be set to the elements stored in ``columns`` joined by
            hashtags, with a hashtag added at the end (e.g. ['a', 'b'] -> 'a#b#').
            - ``output_columns`` should be set to a list of the keys of what's returned from the
            ``get_output_types`` method.
        """

    def test_fit(self):
        """Test the ``fit`` method.

        Setup:
            - mock ``_fit``.

        Input:
            - a dataframe.
            - a list of column names from the dataframe.

        Expected behavior:
            - ``self.columns`` should be set to the passed columns.
            - ``self._fit`` should be called with the correct data (the subset of the
            passed dataframe containing only the passed columns).
            - ``self.output_columns`` should is set correctly.
            - ``column_prefix`` should be set correctly.
        """

    def test_transform_incorrect_columns(self):
        """Test the ``transform`` method when the columns are not in the data.

        When at least on of the passed columns are not present in the data, the method
        should return the data without doing any transformations.

        Setup:
            - set ``self.columns`` to a list of columns, where at least one of them is not
            present in the data

        Input:
            - a dataframe.

        Output:
            - the original data.
        """

    # TODO: should we test all data types being passed? i.e. Series, df and np?
    def test_transform_drop_false(self):
        """Test the ``transform`` method when ``drop=False``.

        Setup:
            - set ``self.columns`` to a list of columns from the data.
            - set `self.output_columns` to the output column names.
            - mock self._transform and return some dummy data.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._transform`` should be called with the correct data.
            - the output columns in the transformed data should match the dummy data
            from ``_tranform``.
            - the set of columns in the transformed data should be correct.
        """

    def test_transform_drop_true(self):
        """Test the ``transform`` method when ``drop=True``.

        Setup:
            - set ``self.columns`` to a list of columns from the data.
            - set `self.output_columns` to the output column names.
            - mock self._transform and return some dummy data.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._transform`` should be called with the correct data.
            - the output columns in the transformed data should match the dummy data
            from ``_tranform``.
            - the set of columns in the transformed data should be correct
            (i.e. since ``drop=True``, input columns are dropped).
        """

    def test_fit_transform(self):
        """Test the ``fit_transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        pass  # TODO: not sure how to test this? just mock fit, transform, and make sure they are each called once?

    def test_reverse_transform_incorrect(self):
        """Test the ``reverse_transform`` method when the columns are not in the data.

        When at least on of the passed columns are not present in ``self.output_columns``, the method
        should return the data without doing any transformations.

        Setup:
            - set ``self.output_columns`` to a list of columns, where at least one of them is not
            present in the data.

        Input:
            - a dataframe.

        Output:
            - the original data.
        """

    def test_reverse_transform_drop_false(self):
        """Test the ``reverse_transform`` method when ``drop=True``.

        Setup:
            - set ``self.output_columns`` to a list of columns from the data.
            - set `self.columns` to the output column names.
            - mock ``self._reverse_transform`` and return some dummy data.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._reverse_transform`` should be called with the correct data.
            - the output columns in the transformed data should match the dummy data
            from ``_reverse_tranform``.
            - the set of columns in the transformed data should be correct
            (i.e. since ``drop=True``, input columns are dropped).
        """

    def test_reverse_transform_drop_true(self):
        """Test the ``reverse_transform`` method when ``drop=False``.

        Setup:
            - set ``self.output_columns`` to a list of columns from the data.
            - set `self.columns` to the output column names.
            - mock ``self._reverese_transform`` and return some dummy data.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._reverse_transform`` should be called with the correct data.
            - the output columns in the transformed data should match the dummy data
            from ``_reverse_tranform``.
            - the set of columns in the transformed data should be correct.
        """
