import abc
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import BaseTransformer, NullTransformer
from rdt.transformers.base import random_state, set_random_states


@patch('rdt.transformers.base.np')
def test_set_random_states(mock_numpy):
    """Test that the method updates the random states correctly.

    The method should do the following steps:
    1. Get the current numpy random state.
    2. Store it.
    3. Switch to the state for the provided function.
    4. Call the provided function.
    5. Set the numpy random state back to what it was.
    """
    # Setup
    initial_state = Mock()
    initial_state_value = Mock()
    initial_state.get_state.return_value = initial_state_value
    random_states = {'fit': initial_state}
    my_function = Mock()
    first_state = Mock()
    second_state = Mock()
    mock_numpy.random.get_state.side_effect = [first_state, second_state]

    # Run
    with set_random_states(random_states, 'fit', my_function):
        pass

    # Assert
    mock_numpy.random.get_state.assert_called()
    mock_numpy.random.set_state.assert_has_calls([
        call(initial_state_value),
        call(first_state)
    ])
    my_function.assert_called_once_with(mock_numpy.random.RandomState.return_value, 'fit')
    mock_numpy.random.RandomState.return_value.set_state.assert_called_with(second_state)


@patch('rdt.transformers.base.set_random_states')
def test_random_state(mock_set_random_states):
    """Test the random_state decorator calls the function and ``set_random_states``.

    The method should create a function that will call ``set_random_states`` and then call
    the passed function within that.
    """
    # Setup
    my_function = Mock()
    my_function.__name__ = 'name'
    instance = Mock()
    instance.random_states = {}
    mock_set_random_state = Mock()
    instance.set_random_state = mock_set_random_state

    # Run
    wrapped_function = random_state(my_function)
    wrapped_function(instance)

    # Assert
    mock_set_random_states.assert_called_once_with({}, 'name', mock_set_random_state)
    my_function.assert_called_once()


@patch('rdt.transformers.base.set_random_states')
def test_random_state_random_states_is_none(mock_set_random_states):
    """Test the random_state decorator calls the function.

    The method should just call the passed function and not ``set_random_states``.
    """
    # Setup
    my_function = Mock()
    instance = Mock()
    instance.random_states = None

    # Run
    wrapped_function = random_state(my_function)
    wrapped_function(instance)

    # Assert
    mock_set_random_states.assert_not_called()
    my_function.assert_called_once()


class TestBaseTransformer:

    def test_set_random_state(self):
        """Test that the method updates the random state for the correct method."""
        # Setup
        transformer = BaseTransformer()
        new_state = Mock()

        # Run
        transformer.set_random_state(new_state, 'fit')

        # Assert
        assert transformer.random_states['fit'] == new_state

    def test_set_random_state_bad_method_name(self):
        """Test that the method raises an error if the passed method is not recognized."""
        # Setup
        transformer = BaseTransformer()
        new_state = Mock()

        # Run
        expected_message = "'method_name' must be one of 'fit', 'transform' or 'reverse_transform'"
        with pytest.raises(ValueError, match=expected_message):
            transformer.set_random_state(new_state, 'fake_method')

    def test_reset_randomization(self):
        """Test that the random seed for ``reverse_transform`` is reset."""
        # Setup
        transformer = BaseTransformer()
        transformer.random_states['fit'] = 0
        transformer.random_states['transform'] = 2
        transformer.random_states['reverse_transform'] = None

        # Run
        transformer.reset_randomization()

        # Assert
        fit_state = transformer.INITIAL_FIT_STATE
        transform_state = transformer.INITIAL_TRANSFORM_STATE
        reverse_transform_state = transformer.INITIAL_REVERSE_TRANSFORM_STATE
        assert transformer.random_states['fit'] == fit_state
        assert transformer.random_states['transform'] == transform_state
        assert transformer.random_states['reverse_transform'] == reverse_transform_state

    def test_get_subclasses(self):
        """Test the ``get_subclasses`` method.

        Validate that any subclass of the ``BaseTransformer`` is returned by the
        ``get_subclasses`` method except if it also inherits from the ``ABC`` class.

        Setup:
            - create a ``Parent`` class which inherits from ``BaseTransformer`` and ``ABC``.
            - create a ``Child`` class which inherits from ``Parent``.

        Output:
            - a list of classes including the ``Child`` class, but NOT including the ``Parent``.
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

    def test_get_input_sdtype(self):
        """Test the ``get_input_sdtype`` method.

        This method should return the value defined in the ``INPUT_SDTYPE`` of the child classes.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``,
            containing only a ``INPUT_SDTYPE`` attribute.

        Output:
            - the string stored in the ``INPUT_SDTYPE`` attribute.
        """
        # Setup
        class Dummy(BaseTransformer):
            INPUT_SDTYPE = 'categorical'

        # Run
        input_sdtype = Dummy.get_input_sdtype()

        # Assert
        assert input_sdtype == 'categorical'

    def test_get_supported_sdtypes_supported_sdtypes(self):
        """Test the ``get_supported_sdtypes`` method.

        This method should return a list with the value defined in the ``SUPPORTED_SDTYPES``
        of the child classes or a list containing the ``INPUT_SDTYPE`` of the child class.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``,
            containing a ``SUPPORTED_SDTYPES`` attribute.

        Output:
            - the list stored in the ``SUPPORTED_SDTYPES`` attribute.
        """
        # Setup
        class Dummy(BaseTransformer):
            SUPPORTED_SDTYPES = ['categorical', 'boolean']

        # Run
        supported_sdtypes = Dummy.get_supported_sdtypes()

        # Assert
        assert supported_sdtypes == ['categorical', 'boolean']

    def test_get_supported_sdtypes_no_supported_sdtypes_provided(self):
        """Test the ``get_supported_sdtypes`` method.

        This method should return a list with the value defined in the ``SUPPORTED_SDTYPES``
        of the child classes or a list containing the ``INPUT_SDTYPE`` of the child class.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``,
            containing a ``INPUT_SDTYPE`` attribute.

        Output:
            - A list with the ``INPUT_SDTYPE`` value inside.
        """
        # Setup
        class Dummy(BaseTransformer):
            INPUT_SDTYPE = 'categorical'

        # Run
        supported_sdtypes = Dummy.get_supported_sdtypes()

        # Assert
        assert supported_sdtypes == ['categorical']

    def test__get_output_to_property(self):
        """Test method adds the column_prefix to output_properties and reformats it."""
        # Setup
        transformer = BaseTransformer()
        transformer.column_prefix = 'abc'
        transformer.output_properties = {
            'col': {'sdtype': 'float', 'next_transformer': None},
            'ignore': {'next_transformer': None},
            None: {'sdtype': 'categorical', 'next_transformer': None}
        }

        # Run
        output = transformer._get_output_to_property('sdtype')

        # Assert
        assert output == {'abc.col': 'float', 'abc': 'categorical'}

    def test___repr___no_parameters(self):
        """Test that the ``__str__`` method returns the class name.

        The ``__repr__`` method should return the class name followed by paranthesis.
        """
        # Setup
        transformer = BaseTransformer()

        # Run
        text = repr(transformer)

        # Assert
        assert text == 'BaseTransformer()'

    def test___repr___with_parameters(self):
        """Test that the ``__repr__`` method returns the class name and parameters.

        The ``_repr__`` method should return the class name followed by all non-default
        parameters wrapped in paranthesis.

        Setup:
            - Create a dummy class which inherits from the ``BaseTransformer`` where:
                - The class has two parameters in its ``__init__`` method with default values.
                - The class instance only sets one of them.
        """
        # Setup
        class Dummy(BaseTransformer):
            def __init__(self, param1=None, param2=None, param3=None):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        transformer = Dummy(param2='value', param3=True)

        # Run
        text = repr(transformer)

        # Assert
        assert text == "Dummy(param2='value', param3=True)"

    def test__str__(self):
        """Test the ``__str__`` method.

        The ``_str__`` method should return the class name followed by all non-default
        parameters wrapped in paranthesis.

        Setup:
            - Create a dummy class which inherits from the ``BaseTransformer`` where:
                - The class has two parameters in its ``__init__`` method with default values.
                - The class instance only sets one of them.
        """
        # Setup
        class Dummy(BaseTransformer):
            def __init__(self, param1=None, param2=None, param3=None):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        transformer = Dummy(param2='value', param3=True)

        # Run
        text = str(transformer)

        # Assert
        assert text == "Dummy(param2='value', param3=True)"

    def test_get_output_sdtypes(self):
        """Test the column_prefix gets added to all columns in output_properties."""
        # Setup
        class Dummy(BaseTransformer):
            column_prefix = 'column_name'

            def __init__(self):
                self.output_properties = {None: {'sdtype': 'numerical'}}

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.get_output_sdtypes()

        # Assert
        assert output == {'column_name': 'numerical'}

    def test_get_next_transformers(self):
        """Test the column_prefix gets added to all columns in output_properties."""
        # Setup
        transformer = NullTransformer()

        class Dummy(BaseTransformer):
            column_prefix = 'column_name'

            def __init__(self):
                self.output_properties = {None: {'next_transformer': transformer}}

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.get_next_transformers()

        # Assert
        assert output == {'column_name': transformer}

    def test_get_input_columns(self):
        """Test the ``get_input_columns method.

        The method should return a list of all the input column names.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``
            and sets the ``columns`` attribute.

        Output:
            - List matching the list created in the setup.
        """
        # Setup
        class Dummy(BaseTransformer):
            columns = ['col1', 'col2', 'col3']

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.get_input_column()

        # Assert
        expected = 'col1'
        assert output == expected

    def test_get_output_columns(self):
        """Test the ``get_output_columns`` method.

        The method should return a list of all the column names created during ``transform``.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``
            and sets the ``column_prefix`` and ``output_properties`` attributes.

        Output:
            - A list of each output name with the prefix prepended.
        """
        # Setup
        class Dummy(BaseTransformer):
            column_prefix = 'column_name'

            def __init__(self):
                self.output_properties = {
                    'out1': {'sdtype': 'numerical'},
                    'out2': {'sdtype': 'float'}
                }

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.get_output_columns()

        # Assert
        expected = ['column_name.out1', 'column_name.out2']
        assert output == expected

    def test_is_generator(self):
        """Test the ``is_generator`` method.

        Validate that this method properly returns the ``IS_GENERATOR`` attribute.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer``, where
            ``IS_GENERATOR`` is set to ``True``.

        Output:
            - the boolean value stored in ``IS_GENERATOR``.
        """
        # Setup
        class Dummy(BaseTransformer):
            IS_GENERATOR = True

        dummy_transformer = Dummy()

        # Run
        output = dummy_transformer.is_generator()

        # Assert
        assert output is True

    def test__store_columns_list(self):
        """Test the ``_store_columns`` method when passed a list.

        When the columns are passed as a list, this method should store the passed columns
        of the data in the ``columns`` attribute.

        Input:
            - a data frame.
            - a list of a subset of the columns of the dataframe.

        Side effects:
            - the ``self.columns`` attribute should be set to the list of the passed columns.
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

        # Assert
        assert base_transformer.columns == ['a', 'b']

    def test__store_columns_tuple(self):
        """Test the ``_store_columns`` method when passed a tuple.

        When the columns are passed as a tuple (and the tuple itself is not a column name), this
        method should store the passed columns of the data in the ``columns`` attribute as a list.

        Input:
            - a data frame.
            - a tuple of a subset of the columns of the dataframe.

        Side effects:
            - the ``self.columns`` attribute should be set to a list of the passed columns.
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
        assert stored_columns == ['a', 'b']

    def test__store_columns_tuple_in_the_data(self):
        """Test the ``_store_columns`` method when passed a tuple which exists in the data.

        When the columns are passed as a tuple and the tuple itself is a column name, it should
        be treated as such, instead of interpreting the elements of the tuple as column names.

        Validate that the stored value in the ``columns`` attribute is a list containing
        the passed tuple.

        Input:
            - a data frame.
            - a tuple which is the name of a column.

        Side effects:
            - the ``self.columns`` attribute should be set to a list containing the passed tuple.
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
        assert stored_columns == [('a', 'b')]

    def test__store_columns_string(self):
        """Test the ``_store_columns`` method when passed a string.

        When the columns are passed as a string, it should be treated as the only column
        name passed and stored in the ``columns`` attribute as a one element list.

        Input:
            - a data frame.
            - a string with the name of one of the columns of the dataframe.

       Side effects:
            - the ``self.columns`` attribute should be set to a list containing the passed string.
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
        assert stored_columns == ['a']

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
        missing = set(columns) - set(data.columns)
        error_msg = f'Columns {missing} were not present in the data.'

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

    def test__add_columns_to_data_series(self):
        """Test the ``_add_columns_to_data`` method.

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
            'b': [4, 5, 6]
        }, index=[2, 0, 1])
        columns = ['c']
        columns_data = pd.Series([7, 8, 9], name='c')

        # Run
        result = BaseTransformer._add_columns_to_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(result, expected)

    def test__add_columns_to_data_dataframe(self):
        """Test the ``_add_columns_to_data`` method.

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
        }, index=[2, 0, 1])
        columns = ['c', 'd']
        columns_data = pd.DataFrame({
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        })

        # Run
        result = BaseTransformer._add_columns_to_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(result, expected)

    def test__add_columns_to_data_1d_array(self):
        """Test the ``_add_columns_to_data`` method.

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
        }, index=[2, 0, 1])
        columns = ['c']
        columns_data = np.array([7, 8, 9], dtype=np.int64)

        # Run
        result = BaseTransformer._add_columns_to_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(result, expected)

    def test__add_columns_to_data_2d_array(self):
        """Test the ``_add_columns_to_data`` method.

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
            'a': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['b', 'c']
        columns_data = np.array([
            [7, 1],
            [8, 5],
            [9, 9]
        ], dtype=np.int64)

        # Run
        result = BaseTransformer._add_columns_to_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [7, 8, 9],
            'c': [1, 5, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(result, expected)

    def test__add_columns_to_data_none(self):
        """Test the ``_add_columns_to_data`` method.

        The method should not change the ``data``.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a ``None``.

        Expected behavior:
            - Data should not be changed.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        }, index=[2, 0, 1])
        columns = []
        columns_data = None

        # Run
        result = BaseTransformer._add_columns_to_data(data, columns_data, columns)

        # Assert
        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(result, expected)

    def test__build_output_columns(self):
        """Test the ``_build_output_columns`` method.

        Validate that the this method stores the correct values in ``self.column_prefix`` and
        ``self.output_columns``.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer`` where:
                - ``columns`` is set to a list of columns from the data.

        Input:
            - a dataframe.

        Side effect:
            - ``self.column_prefix`` should be set to the elements stored in ``self.columns``
            joined by hashtags (e.g. ['a', 'b'] -> 'a#b').
            - ``self.output_columns`` should be set to a list of the keys of what's returned
            from the ``get_output_sdtypes`` method.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']

            def __init__(self):
                self.output_properties = {
                    None: {'sdtype': 'numerical'},
                    'is_null': {'sdtype': 'float'}
                }

        dummy_transformer = Dummy()

        # Run
        dummy_transformer._build_output_columns(data)

        # Assert
        assert dummy_transformer.column_prefix == 'a#b'
        assert dummy_transformer.output_columns == ['a#b', 'a#b.is_null']

    def test__build_output_columns_generated_already_exist(self):
        """Test the ``_build_output_columns`` method.

        When this method generates column names that already exist in the data, it should
        keep adding hashtags at the end of the column name, until the column name becomes unique.
        Under such circumstance, validate that the this method stores the correct values in
        ``column_prefix`` and ``output_columns``.

        Setup:
            - create a ``Dummy`` class which inherits from the ``BaseTransformer`` where:
                - ``columns`` is set to a list of columns from the data.

        Input:
            - a dataframe where the generated column name already exists
            (e.g. ['a', 'b', 'a#b']).

        Side effect:
            - ``self.column_prefix`` should be set to the elements stored in ``self.columns``
            joined by hashtags, with a hashtag added at the end (e.g. ['a', 'b'] -> 'a#b#').
            - ``self.output_columns`` should be set to a list of the keys of what's returned
            from the ``get_output_sdtypes`` method.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'a#b': [4, 5, 6],
            'b': [7, 8, 9],
            'a#b#.is_null': [0, 0, 0],
            'a#b#.is_null#': [0, 0, 0],

        })

        class Dummy(BaseTransformer):
            def __init__(self):
                self.output_properties = {
                    None: {'sdtype': 'numerical'},
                    'is_null': {'sdtype': 'float'}
                }
            columns = ['a', 'b']

        # Run
        dummy_transformer = Dummy()
        dummy_transformer._build_output_columns(data)

        # Assert
        assert dummy_transformer.column_prefix == 'a#b##'
        assert dummy_transformer.output_columns == ['a#b##', 'a#b##.is_null']

    def test__fit_raises_error(self):
        """Test ``_fit`` raises ``NotImplementedError``."""

        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        transformer = BaseTransformer()

        # Run / Assert
        with pytest.raises(NotImplementedError):
            transformer._fit(data)

    def test_fit(self):
        """Test the ``fit`` method.

        Validate that the ``fit`` method (1) sets ``self.columns`` to the passed columns of the
        data, (2) sets ``self.column_prefix`` to the appropriate string (the joined column names
        separated by a hashtag) and (3) sets ``self.output_columns`` to the correct dictionary
        mapping column names to accepted output sdtypes.

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines:
                - a ``_fit`` method which simply stores the passed data to ``self._passed_data``.

        Input:
            - a dataframe.
            - a column name.

        Side effects:
            - ``self.columns`` should be set to the passed columns of the data.
            - ``self.column_prefix`` should be set to the joined column names
            separated by a hashtag.
            - ``self.output_columns`` should be set to the correct dictionary mapping
            column names to accepted output sdtypes.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        column = ['a']

        class Dummy(BaseTransformer):
            def __init__(self):
                super().__init__()
                self.output_properties = {
                    None: {'sdtype': 'categorical'},
                    'is_null': {'sdtype': 'float'}
                }

            def _fit(self, data):
                self._passed_data = data

        dummy_transformer = Dummy()

        # Run
        dummy_transformer.fit(data, column)

        # Assert
        expected_data = pd.Series([1, 2, 3], name='a')
        assert dummy_transformer.columns == ['a']
        pd.testing.assert_series_equal(dummy_transformer._passed_data, expected_data)
        assert dummy_transformer.column_prefix == 'a'
        assert dummy_transformer.output_columns == ['a', 'a.is_null']

    def test__transform_raises_error(self):
        """Test ``_transform`` raises ``NotImplementedError``."""

        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        transformer = BaseTransformer()

        # Run / Assert
        with pytest.raises(NotImplementedError):
            transformer._transform(data)

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

        dummy_transformer = Dummy()

        # Run
        transformed_data = dummy_transformer.transform(data)

        # Assert
        pd.testing.assert_frame_equal(transformed_data, data)

    def test_transform_drop_true(self):
        """Test the ``transform``.

        Validate that the ``transform`` method calls ``self._transform`` with the correct
        data and that the transformed data matches the transformed dummy data (i.e. the original
        data with an added column containing zeros and with the transformed columns dropped).

        Setup:
            - create a dummy class which inherits from the ``BaseTransformer``, which defines:
                - ``columns`` as a list of columns from the data.
                - a ``_transform`` method which stores the passed data to ``self._passed_data``
                and transforms the passed data to a numpy array containing zeros.

        Input:
            - a dataframe.

        Output:
            - the transformed data.

        Side effects:
            - ``self._transform`` should be called with the correct data
            and should store it in ``self._passed_data``.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            output_columns = ['a#b']

            def _transform(self, data):
                self._passed_data = data.copy()
                return np.zeros(len(data))

        dummy_transformer = Dummy()

        # Run
        transformed_data = dummy_transformer.transform(data)

        # Assert
        expected_passed = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_passed)

        expected_transformed = pd.DataFrame({
            'c': [7, 8, 9],
            'a#b': [0.0, 0.0, 0.0],
        })
        pd.testing.assert_frame_equal(transformed_data, expected_transformed)

    def test_fit_transform(self):
        """Test the ``fit_transform`` method.

        Validate that this method calls ``fit`` and ``transform`` once each.

        Setup:
            - create a mock with ``spec_set`` as the ``BaseTransformer``.

        Input:
            - the mock
            - a dataframe.
            - a list of columns from the dataframe.

        Output:
            - the dataframe resulting from fitting and transforming the passed data.

        Side effects:
            - ``fit`` and ``transform`` should each be called once.
        """
        # Setup
        self = Mock(spec_set=BaseTransformer)
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        column = 'a'

        # Run
        output = BaseTransformer.fit_transform(self, data, column)

        # Assert
        self.fit.assert_called_once_with(data, column)
        self.transform.assert_called_once_with(data)
        assert output == self.transform.return_value

    def test__reverse_transform_raises_error(self):
        """Test ``_reverse_transform`` raises ``NotImplementedError``."""

        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        transformer = BaseTransformer()

        # Run / Assert
        with pytest.raises(NotImplementedError):
            transformer._reverse_transform(data)

    def test_reverse_transform_incorrect_columns(self):
        """Test the ``reverse_transform`` method when the columns are not in the data.

        When at least on of the passed columns are not present in ``self.output_columns``,
        the method should return the data without doing any transformations.

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
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            output_columns = ['a', 'b', 'd']

        dummy_transformer = Dummy()

        # Run
        transformed_data = dummy_transformer.reverse_transform(data)

        # Assert
        pd.testing.assert_frame_equal(transformed_data, data)

    def test_reverse_transform(self):
        """Test the ``reverse_transform`` method.

        Validate that the ``reverse_transform`` method calls ``self._reverse_transform``
        with the correct data and that the transformed data matches the transformed dummy data
        (i.e. the original data with an added column containing zeros and the transformed columns
        dropped).
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b.is_null': [4, 5, 6],
            'c': [7, 8, 9]
        })

        class Dummy(BaseTransformer):
            columns = ['a', 'b']
            output_columns = ['a', 'b.is_null']

            def _reverse_transform(self, data):
                self._passed_data = data.copy()
                return np.zeros((len(data), 2))

        # Run
        dummy_transformer = Dummy()
        transformed_data = dummy_transformer.reverse_transform(data)

        # Assert
        expected_passed = pd.DataFrame({
            'a': [1, 2, 3],
            'b.is_null': [4, 5, 6],
        })
        pd.testing.assert_frame_equal(dummy_transformer._passed_data, expected_passed)

        expected_transformed = pd.DataFrame({
            'c': [7, 8, 9],
            'a': [0.0, 0.0, 0.0],
            'b': [0.0, 0.0, 0.0],
        })
        pd.testing.assert_frame_equal(transformed_data, expected_transformed)
