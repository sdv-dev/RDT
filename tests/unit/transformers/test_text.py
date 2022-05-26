"""Test Text Transformers."""

from string import ascii_uppercase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers.null import NullTransformer
from rdt.transformers.text import RegexGenerator


class AsciiGenerator:
    """Ascii Upercase Generator."""

    def __init__(self, max_size=26):
        self.pos = 0
        self.max_size = max_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.max_size:
            raise StopIteration

        char = ascii_uppercase[self.pos]
        self.pos += 1

        return char


class TestRegexGenerator:
    """Test class for ``RegexGenerator``."""

    def test___init__default(self):
        """Test the default instantiation of the transformer.

        Test that ``RegexGenerator`` defaults to ``regex_format='[A-Za-z]{5}'``

        Side effects:
            - the ``instance.missing_value_replacement`` is ``None``.
            - the ``instance.model_missing_values`` is ``False``
            - the ``instance.regex_format`` is ``'[A-Za-z]{5}'``'.
        """
        # Run
        instance = RegexGenerator()

        # Assert
        assert instance.data_length is None
        assert instance.missing_value_replacement is None
        assert instance.model_missing_values is False
        assert instance.regex_format == '[A-Za-z]{5}'

    def test___init__custom(self):
        """Test the default instantiation of the transformer.

        Test that when creating an instance of ``RegexGenerator`` and passing a
        ``regex_format`` this is being stored and the ``missing_value_replacement`` or
        ``model_missing_values`` are set to the custom values as well.

        Side effects:
            - the ``instance.missing_value_replacement`` is ``AAAA``.
            - the ``instance.model_missing_values`` is ``True``
            - the ``instance.regex_format`` is ``'[A-Za-z]{5}'``'.
        """
        # Run
        instance = RegexGenerator(
            regex_format='[0-9]',
            missing_value_replacement='AAAA',
            model_missing_values=True
        )

        # Assert
        assert instance.data_length is None
        assert instance.missing_value_replacement == 'AAAA'
        assert instance.model_missing_values
        assert instance.regex_format == '[0-9]'

    def test_get_output_sdtypes(self):
        """Test the ``get_output_sdtypes``.

        Setup:
            - initialize a ``RegexGenerator`` transformer with default values.

        Output:
            - the ``output_sdtypes`` returns an empty dictionary.
        """
        # Setup
        instance = RegexGenerator()
        instance.column_prefix = 'a#b'

        # Run
        output = instance.get_output_sdtypes()

        # Assert
        expected = {}
        assert output == expected

    def test_get_output_sdtypes_model_missing_values(self):
        """Test the ``get_output_sdtypes`` method when a null column is created.

        Setup:
            - initialize a ``RegexGenerator`` transformer which:
                - sets ``self.null_transformer`` to a ``NullTransformer`` where
                ``self.model_missing_values`` is True.
                - sets ``self.column_prefix`` to a string.

        Output:
            - An ``output_sdtypes`` dictionary is being returned with the ``self.column_prefix``
              added to the beginning of the keys.
        """
        # Setup
        instance = RegexGenerator()
        instance.null_transformer = NullTransformer(missing_value_replacement='fill')
        instance.null_transformer._model_missing_values = True
        instance.column_prefix = 'a#b'

        # Run
        output = instance.get_output_sdtypes()

        # Assert
        expected = {
            'a#b.is_null': 'float'
        }
        assert output == expected

    @patch('rdt.transformers.text.NullTransformer')
    def test__fit(self, mock_null_transformer):
        """Test the ``_fit`` method.

        Validate that the ``_fit`` method uses the ``NullTransformer`` to parse the data
        and learn the length of it.

        Setup:
            - Initialize a ``RegexGenerator`` transformer.
            - Mock the ``NullTransformer``.

        Input:
            - ``pd.Series`` containing 3 strings.

        Side Effects:
            - ``NullTransformer`` instance has been created with ``model_missing_values`` as
              ``False`` and ``missing_value_replacement`` as ``None``.
            - ``ǸullTransformer`` instance method ``fit`` has been called with the input data.
            - ``instance.data_length`` equals to the length of the input data.
        """
        # Setup
        instance = RegexGenerator()

        columns_data = pd.Series(['1', '2', '3'])

        # Run
        instance._fit(columns_data)

        # Assert
        mock_null_transformer.assert_called_once_with(None, False)
        mock_null_transformer.return_value.fit.called_once_with(columns_data)
        assert instance.data_length == 3

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method returns ``None`` when the ``NullTransformer``
        does not model the missing values.

        Setup:
            - Initialize a ``RegexGenerator`` transformer.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - ``None``.
        """
        # Setup
        columns_data = pd.Series([1, 2, 3])
        instance = RegexGenerator()

        # Run
        result = instance._transform(columns_data)

        # Assert
        assert result is None

    def test__transform_model_missing_values(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method uses the ``NullTransformer`` instance to
        transform the data.

        Setup:
            - Initialize a ``RegexGenerator`` transformer.
            - Mock the ``null_transformer`` of the instance.
            - Mock the return value of the ``null_transformer.transform``.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - The second dimension of the mocked return value of the
              ``null_transformer.transform``.
        """
        # Setup
        columns_data = pd.Series([1, 2, 3])
        instance = RegexGenerator()
        instance.null_transformer = Mock()

        instance.null_transformer.transform.return_value = np.array([
            [4, 0],
            [5, 1],
            [6, 0],
        ])

        # Run
        result = instance._transform(columns_data)

        # Assert
        instance.null_transformer.transform.assert_called_once_with(columns_data)
        np.testing.assert_array_equal(result, np.array([0, 1, 0]))

    @patch('rdt.transformers.text.strings_from_regex')
    def test__reverse_transform_generator_size_bigger_than_data_length(self,
                                                                       mock_strings_from_regex):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``strings_from_regex``
        function using the ``instance.regex_format`` and then generates the
        ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 4.
            - Initialize a generator.

        Mock:
            - Mock the ``strings_from_regex`` function to return a generator and a size of 5.
            - Mock the ``null_transformer`` of the instance.
            - Mock the return value of the ``null_transformer.reverse_transform``.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series()
        instance.null_transformer = Mock()
        instance.null_transformer.models_missing_values.return_value = False
        instance.data_length = 3
        generator = AsciiGenerator(max_size=5)
        mock_strings_from_regex.return_value = (generator, 5)
        instance.null_transformer.reverse_transform.return_value = np.array(['A', 'B', 'C'])

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_null_call = np.array(['A', 'B', 'C'])
        null_call = instance.null_transformer.reverse_transform.call_args_list[0][0][0]
        np.testing.assert_array_equal(null_call, expected_null_call)
        np.testing.assert_array_equal(result, np.array(['A', 'B', 'C']))

    @patch('rdt.transformers.text.strings_from_regex')
    def test__reverse_transform_generator_size_smaller_than_data_length(self,
                                                                        mock_strings_from_regex):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``strings_from_regex``
        function using the ``instance.regex_format`` and then generates the
        ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 4.
            - Initialize a generator.

        Mock:
            - Mock the ``strings_from_regex`` function to return a generator and a size of 5.
            - Mock the ``null_transformer`` of the instance.
            - Mock the return value of the ``null_transformer.reverse_transform``.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series()
        instance.null_transformer = Mock()
        instance.null_transformer.models_missing_values.return_value = False
        instance.data_length = 11
        generator = AsciiGenerator(5)
        mock_strings_from_regex.return_value = (generator, 5)
        return_value = np.array(['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A'])
        instance.null_transformer.reverse_transform.return_value = return_value
        instance.columns = ['a']

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_null_call = np.array(['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A'])

        null_call = instance.null_transformer.reverse_transform.call_args_list[0][0][0]
        np.testing.assert_array_equal(null_call, expected_null_call)
        np.testing.assert_array_equal(result, expected_null_call)

    @patch('rdt.transformers.text.warnings')
    @patch('rdt.transformers.text.strings_from_regex')
    def test__reverse_transform_models_missing_values(self,
                                                      mock_strings_from_regex, mock_warnings):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``strings_from_regex``
        function using the ``instance.regex_format`` and then generates the
        ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 4.
            - Initialize a generator.

        Mock:
            - Mock the ``strings_from_regex`` function to return a generator and a size of 5.
            - Mock the ``null_transformer`` of the instance.
            - Mock the return value of the ``null_transformer.reverse_transform``.
            - Mock warnings and assert that has been called once

        """
        # Setup
        columns_data = pd.DataFrame({'is_null': [0, 1, 0, 0, 0, 0]})
        instance = RegexGenerator('[A-Z]')
        instance.null_transformer = Mock()
        instance.null_transformer.models_missing_values.return_value = True
        instance.data_length = 6
        generator = AsciiGenerator(5)
        mock_strings_from_regex.return_value = (generator, 5)
        return_value = np.array(['A', np.nan, 'C', 'D', 'E', 'A'])
        instance.columns = ['a']
        instance.null_transformer.reverse_transform.return_value = return_value

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_warning_message = (
            "The data has 6 rows but the regex for 'a' "
            'can only create 5 unique values. Some values in '
            "'a' may be repeated."
        )
        expected_null_call = np.array([
            ['A', 0],
            ['B', 1],
            ['C', 0],
            ['D', 0],
            ['E', 0],
            ['A', 0]
        ], dtype=object)
        expected_result = np.array(['A', np.nan, 'C', 'D', 'E', 'A'])

        null_call = instance.null_transformer.reverse_transform.call_args_list[0][0][0]
        np.testing.assert_array_equal(null_call, expected_null_call)
        np.testing.assert_array_equal(result, expected_result)

        mock_warnings.warn.assert_called_once_with(expected_warning_message)
