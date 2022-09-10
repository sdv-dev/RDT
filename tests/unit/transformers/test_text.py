"""Test Text Transformers."""

import re
from string import ascii_uppercase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt.errors import Error
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
            - the ``instance.regex_format`` is ``'[A-Za-z]{5}'``'.
        """
        # Run
        instance = RegexGenerator()

        # Assert
        assert instance.data_length is None
        assert instance.regex_format == '[A-Za-z]{5}'
        assert instance.enforce_uniqueness is False

    def test___init__custom(self):
        """Test the default instantiation of the transformer.

        Test that when creating an instance of ``RegexGenerator`` and passing a
        ``regex_format`` this is being stored.

        Side effects:
            - the ``instance.regex_format`` is ``'[A-Za-z]{5}'``'.
            - ``instance.enforce_uniqueness`` is ``True``.
        """
        # Run
        instance = RegexGenerator(
            regex_format='[0-9]',
            enforce_uniqueness=True
        )

        # Assert
        assert instance.data_length is None
        assert instance.regex_format == '[0-9]'
        assert instance.enforce_uniqueness

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that the ``_fit`` method learns the original data length.

        Setup:
            - Initialize a ``RegexGenerator`` transformer.

        Input:
            - ``pd.Series`` containing 3 strings.

        Side Effects:
            - ``instance.data_length`` equals to the length of the input data.
        """
        # Setup
        instance = RegexGenerator()
        columns_data = pd.Series(['1', '2', '3'])

        # Run
        instance._fit(columns_data)

        # Assert
        assert instance.data_length == 3

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method returns ``None``.

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

    @patch('rdt.transformers.text.strings_from_regex')
    def test__reverse_transform_generator_size_bigger_than_data_length(self,
                                                                       mock_strings_from_regex):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``strings_from_regex``
        function using the ``instance.regex_format`` and then generates the
        ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 3.
            - Initialize a generator.

        Mock:
            - Mock the ``strings_from_regex`` function to return a generator and a size of 5.

        Output:
            - A ``numpy.array`` with the first three letters from the generator.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series()
        instance.data_length = 3
        generator = AsciiGenerator(max_size=5)
        mock_strings_from_regex.return_value = (generator, 5)

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
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
            - Set ``data_length`` to 11.
            - Initialize a generator.

        Mock:
            - Mock the ``strings_from_regex`` function to return a generator and a size of 5.

        Output:
            - A ``numpy.array`` with the first five letters from the generator repeated.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series()
        instance.data_length = 11
        generator = AsciiGenerator(5)
        mock_strings_from_regex.return_value = (generator, 5)
        instance.columns = ['a']

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_result = np.array(['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A'])
        np.testing.assert_array_equal(result, expected_result)

    @patch('rdt.transformers.text.strings_from_regex')
    def test__reverse_transform_generator_size_of_input_data(self, mock_strings_from_regex):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``strings_from_regex``
        function using the ``instance.regex_format`` and then generates the
        ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 2.
            - Initialize a generator.

        Input:
            - ``pandas.Series`` with a length of ``4``.
        Mock:
            - Mock the ``strings_from_regex`` function to return a generator and a size of 5.

        Output:
            - A ``numpy.array`` with the first five letters from the generator repeated.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series([1, 2, 3, 4])
        instance.data_length = 2
        generator = AsciiGenerator(5)
        mock_strings_from_regex.return_value = (generator, 5)
        instance.columns = ['a']

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_result = np.array(['A', 'B', 'C', 'D'])
        np.testing.assert_array_equal(result, expected_result)
