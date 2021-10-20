import abc
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.base import BaseTransformer


class TestBaseTransformer:

    def test_get_subclasses(self):
        """Test the ``get_subclasses`` method.

        Setup:
            - create a ``Parent`` class which inherits from ``BaseTransformer`` and ``ABC``.
            - create a ``Child`` class which inherits from ``Parent``.

        Output:
            - the list of child classes which inherit from ``BaseTransformer``.

        Expected behavior:
            - the subclasses should include the ``Child`` class but not the ``Parent`` class.
        """
        # Setup
        class Parent(BaseTransformer, abc.ABC):
            pass
        class Child(Parent):
            pass
        
        # Run
        subclasses = BaseTransformer.get_subclasses()

        # Assert
        assert Child in subclasses
        assert Parent not in subclasses

    def test_get_input_type(self):
        """Test the ``get_input_type`` method.

        This method should return the value defined in the ``INPUT_TYPE`` of the child classes.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``,
            containing only a ``INPUT_TYPE`` attribute.

        Output:
            - the string stored in the ``INPUT_TYPE`` attribute.
        """
        # Setup
        class Dummy(BaseTransformer):
            INPUT_TYPE = 'categorical'
        
        # Run
        input_type = Dummy.get_input_type()

        # Assert
        assert input_type == 'categorical'

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
        assert output is None

    def test__add_prefix_dictionary(self):
        """Test the ``_add_prefix`` method when passed a dictionary.

        Setup:
            - set the ``column_prefix`` of the ``BaseTransformer`` to ``'column_name'``.

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
            - create a ``Dummy`` class which inherits from the ``BaseTransformer`` where:
                - ``column_prefix`` is set to ``'colulumn_name'``.
                - ``OUTPUT_TYPES`` is set to dictionary.

        Output:
            - the dictionary set in ``OUTPUT_TYPES`` with the ``column_prefix`` string
            added to the beginning of the keys.
        """
        # Setup
        class Dummy(BaseTransformer):
            column_prefix = 'column_name'
            OUTPUT_TYPES = {
                'value': 'numerical'
            }
        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.get_output_types()

        # Assert
        expected = {
            'column_name.value': 'numerical'
        }
        assert output == expected

    def test_is_transform_deterministic(self):
        """Test the ``is_transform_deterministic`` method.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``, where
            ``DETERMINISTIC_TRANSFORM`` is set to True.

        Output:
            - the boolean value stored in ``DETERMINISTIC_TRANSFORM``.
        """
        # Setup
        class Dummy(BaseTransformer):
            DETERMINISTIC_TRANSFORM = True

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.is_transform_deterministic()

        # Assert
        assert output == True

    def test_is_reverse_deterministic(self):
        """Test the ``is_reverse_deterministic`` method.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``, where
            ``DETERMINISTIC_REVERSE`` is set to True.

        Output:
            - the boolean value stored in ``DETERMINISTIC_REVERSE``.
        """
        # Setup
        class Dummy(BaseTransformer):
            DETERMINISTIC_REVERSE = True

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.is_reverse_deterministic()

        # Assert
        assert output == True

    def test_is_composition_identity(self):
        """Test the ``is_composition_identity`` method.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``, where
            ``COMPOSITION_IS_IDENTITY`` is set to True.

        Output:
            - the boolean value stored in ``COMPOSITION_IS_IDENTITY``.
        """
        # Setup
        class Dummy(BaseTransformer):
            COMPOSITION_IS_IDENTITY = True

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.is_composition_identity()

        # Assert
        assert output == True

    def test_get_next_transformers(self):
        """Test the ``get_next_transformers`` method.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer`` where:
                - ``column_prefix`` is set to a string.
                - ``NEXT_TRANSFORMERS`` is set to dictionary.

        Output:
            - the dictionary set in ``NEXT_TRANSFORMERS`` with the ``column_prefix`` string
            added to the beginning of the keys.
        """
        # Setup
        class Dummy(BaseTransformer):
            column_prefix = 'column_name'
            NEXT_TRANSFORMERS = {
                'value': 'NullTransformer'
            }
        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.get_next_transformers()

        # Assert
        expected = {
            'column_name.value': 'NullTransformer'
        }
        assert output == expected

    def test__store_columns_list(self):
        """Test the ``_store_columns`` method when passed a list.

        Input:
            - a data frame.
            - a list of a subset of the columns of the dataframe.
        
        Expected behavior:
            - the method should store the passed columns in ``self.columns``.
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

        Input:
            - a data frame.
            - a tuple of a subset of the columns of the dataframe.
        
        Expected behavior:
            - the method should first convert the tuple to list, an then store
            the passed columns in ``self.columns``.
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

    def test__store_columns_tuple_in_the_data(self):
        """Test the ``_store_columns`` method when passed a tuple which exists in the data.

        When the passed tuple is the name of one of the columns of the data, it should be
        treated as such, instead of interpreting the elements of the tuple as column names. 

        Input:
            - a data frame.
            - a tuple which is the name of a column.
        
        Expected behavior:
            - the method should first convert add the tuple to a list,
            and then store this one column in ``self.columns``.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            ('a', 'b'): [7, 8, 9]
        })
        columns = ('a', 'b')
        base_transformer = BaseTransformer()

        # Run
        base_transformer._store_columns(columns, data)
        stored_columns = base_transformer.columns

        # Assert
        expected = [('a', 'b')]
        assert stored_columns == expected

    def test__store_columns_string(self):
        """Test the ``_store_columns`` method when passed a string.

        Input:
            - a data frame.
            - a string with the name of one of the columns of the dataframe.

        Expected behavior:
            - the method should first add the passed string to a list,
            and then store this one column in ``self.columns``.
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

    def test__set_columns_series(self):
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
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['c']
        columns_data = pd.Series([7, 8, 9], name='c')

        # Run
        BaseTransformer._set_columns_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__set_columns_data_dataframe(self):
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
        # Setup
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

        # Run
        BaseTransformer._set_columns_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__set_columns_1d_array(self):
        """Test the ``_set_columns_data`` method.

        The method should not reorder the rows from the ``columns_data``
        parameter if it is a 1d array and the ``data`` has a different index.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a 1d array.
            - columns will have the column name of the array.

        Expected behavior:
            - Data should have the values from columns_data in the same order
            as they were in columns_data.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['c']
        columns_data = np.array([7, 8, 9])

        # Run
        BaseTransformer._set_columns_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__set_columns_2d_array(self):
        """Test the ``_set_columns_data`` method.

        The method should not reorder the rows from the ``columns_data``
        parameter if it is a ``Series`` and the ``data`` has a different index.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a 2d array with a sequential index.
            - columns will have the column name of the 2d array.

        Expected behavior:
            - Data should have the values from columns_data in the same order
            as they were in columns_data.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['b', 'c']
        columns_data = np.array([
            [7, 1],
            [8, 5],
            [9, 9]
        ])

        # Run
        BaseTransformer._set_columns_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [7, 8, 9],
            'c': [1, 5, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__build_output_columns(self):
        """Test the ``_build_output_columns`` method.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer`` where:
                - ``columns`` is set to a list of columns from the data.
                - ``OUTPUT_TYPES`` is set to a dictionary.

        Input:
            - a dataframe.

        Expected behavior:
            - ``column_prefix`` should be set to the elements stored in ``columns`` joined by
            hashtags (e.g. ['a', 'b'] -> 'a#b').
            - ``output_columns`` should be set to a list of the keys of what's returned from the
            ``get_output_types`` method.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            OUTPUT_TYPES = {
                'value': 'numerical',
                'is_null': 'float'
            }

        # Run
        dummy_transformer = Dummy()
        dummy_transformer._build_output_columns(data)

        # Assert
        assert dummy_transformer.column_prefix == 'a#b'
        assert dummy_transformer.output_columns == ['a#b.value', 'a#b.is_null']

    def test__build_output_columns_generated_already_exist(self):
        """Test the ``_build_output_columns`` method.

        When this method generates column names that already exist in the data, it should
        keep adding hashtags at the end of the column name, until the column name becomes unique.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer`` where:
                - ``columns`` is set to a list of columns from the data.
                - ``OUTPUT_TYPES`` is set to a dictionary.

        Input:
            - a dataframe where the generated column name already exists
            (e.g. ['a', 'b', 'a#b.value']).

        Expected behavior:
            - ``column_prefix`` should be set to the elements stored in ``columns`` joined by
            hashtags, with a hashtag added at the end (e.g. ['a', 'b'] -> 'a#b#').
            - ``output_columns`` should be set to a list of the keys of what's returned from the
            ``get_output_types`` method.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'a#b.value': [4, 5, 6],
            'b': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            OUTPUT_TYPES = {
                'value': 'numerical',
                'is_null': 'float'
            }
            columns = ['a', 'b']

        # Run
        dummy_transformer = Dummy()
        dummy_transformer._build_output_columns(data)

        # Assert
        assert dummy_transformer.column_prefix == 'a#b#'
        assert dummy_transformer.output_columns == ['a#b#.value', 'a#b#.is_null']

    def test_fit(self):
        """Test the ``fit`` method.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines:
                - a ``OUTPUT_TYPES`` dictionary.
                - a ``_fit`` method which simply stores the passed data to ``self._passed_data``.

        Input:
            - a dataframe.
            - a list of column names from the dataframe.

        Expected behavior:
            - ``self.columns`` should be set to the passed columns.
            - ``self._fit`` should be called with the correct data (the subset of the
            passed dataframe containing only the passed columns).
            - ``self.output_columns`` should is set correctly.
            - ``self.column_prefix`` should be set correctly.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        columns = ['a', 'b']

        class Dummy(BaseTransformer):
            OUTPUT_TYPES = {
                'value': 'categorical',
                'is_null': 'float'
            }
            def _fit(self, data):
                self._passed_data = data

        # Run
        dummy_transformer = Dummy()
        dummy_transformer.fit(data, columns)

        # Assert
        expected_data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        assert dummy_transformer.columns == ['a', 'b']
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_data)
        assert dummy_transformer.column_prefix == 'a#b'
        assert dummy_transformer.output_columns == ['a#b.value', 'a#b.is_null']

    def test_transform_incorrect_columns(self):
        """Test the ``transform`` method when the columns are not in the data.

        When at least one of the passed columns are not present in the data, the method
        should return the data without doing any transformations.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines
            ``columns`` as list of columns where at least one of them is not present in the data

        Input:
            - a dataframe.

        Output:
            - the original data.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b', 'd']

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.transform(data)

        # Assert
        pd.testing.assert_frame_equal(transformed_data, data)

    def test_transform_drop_false(self):
        """Test the ``transform`` method when ``drop=False``.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines:
                - ``columns`` as a list of columns from the data.
                - a ``_transform`` method which stores the passed data to ``self._passed_data``
                and transforms the passed data to a numpy array containing zeros.

        Input:
            - a dataframe.
            - drop = False.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._transform`` should be called with the correct data.
            - the transformed data should match the dummy data,
            but with an added column of zeros.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            output_columns = ['a#b.value']
            def _transform(self, data):
                self._passed_data = data.copy()
                return np.zeros(len(data))

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.transform(data, drop=False)

        # Assert        
        expected_passed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_passed)

        expected_transformed  = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'a#b.value': [0.0, 0.0, 0.0],
        })
        pd.testing.assert_frame_equal(transformed_data, expected_transformed)

    def test_transform_drop_true(self):
        """Test the ``transform`` method when ``drop=True``.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines:
                - ``columns`` as a list of columns from the data.
                - a ``_transform`` method which stores the passed data to ``self._passed_data``
                and transforms the passed data to a numpy array containing zeros.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._transform`` should be called with the correct data.
            - the transformed data should match the dummy data, but with an added column
            of zeros and with the transformed columns dropped.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            output_columns = ['a#b.value']
            def _transform(self, data):
                self._passed_data = data.copy()
                return np.zeros(len(data))

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.transform(data)

        # Assert
        expected_passed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_passed)
        expected_transformed  = pd.DataFrame({
            'c': [7, 8, 9],
            'a#b.value': [0.0, 0.0, 0.0],
        })
        pd.testing.assert_frame_equal(transformed_data, expected_transformed)

    def test_reverse_transform_incorrect_columns(self):
        """Test the ``reverse_transform`` method when the columns are not in the data.

        When at least on of the passed columns are not present in ``self.output_columns``, the method
        should return the data without doing any transformations.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines
            ``columns`` as list of columns where at least one of them is not present in the data.

        Input:
            - a dataframe.

        Output:
            - the original data.
        """
        # Setup
        data = pd.DataFrame({
            'a.value': [1, 2, 3],
            'b.value': [4, 5, 6],
            'c.value': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            output_columns = ['a.value', 'b.value', 'd.value']

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.reverse_transform(data)

        # Assert
        pd.testing.assert_frame_equal(transformed_data, data)

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
            - the transformed data should match the dummy data, but with added columns
            of zeros for the reverse transformed columns.
        """
        # Setup
        data = pd.DataFrame({
            'a.value': [1, 2, 3],
            'b.value': [4, 5, 6],
            'c.value': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            output_columns = ['a.value', 'b.value']
            def _reverse_transform(self, data):
                self._passed_data = data.copy()
                return np.zeros((len(data), 2))

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.reverse_transform(data, drop=False)

        # Assert
        expected_passed = pd.DataFrame({
            'a.value': [1, 2, 3],
            'b.value': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_passed)
        
        expected_transformed = pd.DataFrame({
            'a.value': [1, 2, 3],
            'b.value': [4, 5, 6],
            'c.value': [7, 8, 9],
            'a': [0.0, 0.0, 0.0],
            'b': [0.0, 0.0, 0.0]
        })
        pd.testing.assert_frame_equal(transformed_data, expected_transformed)

    def test_reverse_transform_drop_true(self):
        """Test the ``reverse_transform`` method when ``drop=False``.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines:
                - ``columns`` as a list of columns from the data.
                - a ``_reverse_transform`` method which stores the passed data to
                ``self._passed_data`` and transforms the passed data to a numpy array
                containing zeros.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Expected behavior:
            - ``self._reverse_transform`` should be called with the correct data.
            - the transformed data should match the dummy data, but with the added columns
            of zeros and with the transformed columns dropped.
        """
        # Setup
        data = pd.DataFrame({
            'a.value': [1, 2, 3],
            'b.value': [4, 5, 6],
            'c.value': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            output_columns = ['a.value', 'b.value']
            def _reverse_transform(self, data):
                self._passed_data = data.copy()
                return np.zeros((len(data), 2))

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.reverse_transform(data)

        # Assert
        expected_passed = pd.DataFrame({
            'a.value': [1, 2, 3],
            'b.value': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_passed)
        
        expected_transformed = pd.DataFrame({
            'c.value': [7, 8, 9],
            'a': [0.0, 0.0, 0.0],
            'b': [0.0, 0.0, 0.0],
        })
        pd.testing.assert_frame_equal(transformed_data, expected_transformed)
