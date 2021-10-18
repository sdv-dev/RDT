from unittest.mock import patch

import pandas as pd
import pytest

from rdt.transformers.base import BaseTransformer


class TestBaseTransformer:

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

    def test_get_subclasses(self):
        """Test the ``get_subclasses`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        pass

    def test_get_input_type(self):
        """Test the ``get_input_type`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        with patch.object(BaseTransformer, 'INPUT_TYPE', 'categorical'):
            input_type = BaseTransformer.get_input_type()
            expected = 'categorical'

            assert input_type == expected

    def test__add_prefix_none(self):
        """Test the ``_add_prefix`` method when passed a ``None``.

        The method should simply return ``None`` when ``None`` is passed as an argument.

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
        base_transformer = BaseTransformer()
        base_transformer.column_prefix = 'column_name'
        base_transformer.NEXT_TRANSFORMERS = {
            'value': 'NullTransformer'
        }
        output = base_transformer.get_next_transformers()

        expected = {
            'column_name.value': 'NullTransformer'
        }
        assert output == expected

    def test__store_columns_list(self):
        """Test the ``_store_columns`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['a', 'b']

        base_transformer = BaseTransformer()
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns
        expected = ['a', 'b']

        assert stored_columns == expected

    def test__store_columns_tuple(self):
        """Test the ``_store_columns`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ('a', 'b')

        base_transformer = BaseTransformer()
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns
        expected = ['a', 'b']

        assert stored_columns == expected

    def test__store_columns_string(self):
        """Test the ``_store_columns`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = 'a'

        base_transformer = BaseTransformer()
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns
        expected = ['a']

        assert stored_columns == expected

    def test__store_columns_missing(self):
        """Test the ``_store_columns`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        data = pd.DataFrame()
        columns = ['a', 'b']
        base_transformer = BaseTransformer()

        missing = set(columns)
        error_msg = (f'Columns {missing} were not present in the data.')

        with pytest.raises(KeyError, match=error_msg):
            base_transformer._store_columns(columns, data)

    def test__get_columns_data_multiple_columns(self):
        """Test the ``_get_columns_data`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['a', 'b']

        columns_data = BaseTransformer._get_columns_data(data, columns)

        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(columns_data, expected)

    def test__get_columns_data_single_column(self):
        """Test the ``_get_columns_data`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['b']

        columns_data = BaseTransformer._get_columns_data(data, columns)

        expected = pd.Series([4, 5, 6], name='b')
        pd.testing.assert_series_equal(columns_data, expected)

    def test__build_output_columns(self):
        """Test the ``_build_output_columns`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """

    def test_fit(self):
        """Test the ``fit`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """

    def test_transform_incorrect_columns(self):
        """Test the ``transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """

    def test_transform_drop_true(self):
        """Test the ``transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """

    def test_transform_drop_false(self):
        """Test the ``transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
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

    def test_reverse_transform_incorrect(self):
        """Test the ``reverse_transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """

    def test_reverse_transform_drop_true(self):
        """Test the ``reverse_transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """

    def test_reverse_transform_drop_false(self):
        """Test the ``reverse_transform`` method.

        Setup:
            -

        Input:
            -

        Expected behavior:
            -
        """
