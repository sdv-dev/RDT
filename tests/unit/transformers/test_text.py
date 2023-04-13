"""Test Text Transformers."""

import re
from string import ascii_uppercase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt.errors import TransformerProcessingError
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

    def test___getstate__(self):
        """Test that ``__getstate__`` returns a dictionary without the generator."""
        # Setup
        instance = RegexGenerator()
        instance.reset_randomization()
        mock_random_sates = Mock()
        instance.random_states = mock_random_sates

        # Run
        state = instance.__getstate__()

        # Assert
        assert state == {
            'data_length': None,
            'enforce_uniqueness': False,
            'generated': 0,
            'generator_size': 380204032,
            'output_properties': {None: {'next_transformer': None}},
            'regex_format': '[A-Za-z]{5}',
            'random_states': mock_random_sates,
        }

    @patch('rdt.transformers.text.strings_from_regex')
    def test___setstate__generated_and_generator_size(self, mock_strings_from_regex):
        """Test that ``__setstate__`` will initialize a generator and wind it forward."""
        # Setup
        state = {
            'data_length': None,
            'enforce_uniqueness': False,
            'generated': 10,
            'generator_size': 380204032,
            'output_properties': {None: {'next_transformer': None}},
            'regex_format': '[A-Za-z]{5}'
        }
        generator = AsciiGenerator()
        mock_strings_from_regex.return_value = (generator, 26)
        instance = RegexGenerator()

        # Run
        instance.__setstate__(state)

        # Assert
        assert next(generator) == 'K'
        assert instance.generated == 10
        assert instance.generator_size == 380204032
        mock_strings_from_regex.assert_called_once_with('[A-Za-z]{5}')

    @patch('rdt.transformers.text.strings_from_regex')
    def test___setstate__(self, mock_strings_from_regex):
        """Test that ``__setstate__`` will initialize a generator but not forward it.

        When ``generated`` is ``None`` and ``generator_size`` is ``None`` this will be assigned
        the ``0`` and the ``generator_size`` respectively.
        """
        # Setup
        state = {
            'data_length': None,
            'enforce_uniqueness': False,
            'generated': None,
            'generator_size': None,
            'output_properties': {None: {'next_transformer': None}},
            'regex_format': '[A-Za-z]{5}'
        }
        generator = AsciiGenerator()
        mock_strings_from_regex.return_value = (generator, 26)
        instance = RegexGenerator()

        # Run
        instance.__setstate__(state)

        # Assert
        assert next(generator) == 'A'
        assert instance.generated == 0
        assert instance.generator_size == 26
        mock_strings_from_regex.assert_called_once_with('[A-Za-z]{5}')

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

    @patch('rdt.transformers.text.BaseTransformer.reset_randomization')
    @patch('rdt.transformers.text.strings_from_regex')
    def test_reset_randomization(self, mock_strings_from_regex, mock_base_reset):
        """Test that this method creates a new generator.

        This method should create a new ``instance.generator``, ``instance.generator_size`` and
        restart the ``instance.generated`` values to 0.
        """
        # Setup
        generator = AsciiGenerator(5)
        mock_strings_from_regex.return_value = (generator, 2)
        instance = RegexGenerator()

        # Run
        instance.reset_randomization()

        # Assert
        assert instance.generator == generator
        assert instance.generator_size == 2
        assert instance.generated == 0
        mock_strings_from_regex.assert_called_once_with('[A-Za-z]{5}')
        mock_base_reset.assert_called_once()

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
        assert instance.output_properties == {None: {'next_transformer': None}}

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

    def test__reverse_transform_generator_size_bigger_than_data_length(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method uses the ``instance.generator``
        to generate the ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 3.
            - Initialize a generator.
            - Set a generator, generator size and generated values.

        Output:
            - A ``numpy.array`` with the first three letters from the generator.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series()
        instance.data_length = 3
        generator = AsciiGenerator(max_size=5)
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        np.testing.assert_array_equal(result, np.array(['A', 'B', 'C']))

    def test__reverse_transform_generator_size_smaller_than_data_length(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method uses the ``instance.generator``
        to generate the ``instance.data_length`` number of data when ``enforce_uniqueness`` is
        ``False`` but the data to be created is bigger.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 11.
            - Initialize a generator.

        Output:
            - A ``numpy.array`` with the first five letters from the generator repeated.
        """
        # Setup
        instance = RegexGenerator('[A-Z]', enforce_uniqueness=False)
        columns_data = pd.Series()
        instance.reset_randomization = Mock()
        instance.data_length = 11
        generator = AsciiGenerator(5)
        instance.columns = ['a']
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_result = np.array(['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A'])
        np.testing.assert_array_equal(result, expected_result)

    def test__reverse_transform_generator_size_of_input_data(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method uses the ``instance.generator``
        to generate the ``instance.data_length`` number of data when ``enforce_uniqueness`` is
        ``False`` but the data to be created is bigger.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 2.
            - Initialize a generator.

        Input:
            - ``pandas.Series`` with a length of ``4``.

        Output:
            - A ``numpy.array`` with the first five letters from the generator repeated.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series([1, 2, 3, 4])
        instance.data_length = 2
        generator = AsciiGenerator(5)
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0
        instance.columns = ['a']

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        expected_result = np.array(['A', 'B', 'C', 'D'])
        np.testing.assert_array_equal(result, expected_result)
        assert instance.generated == 4

    def test__reverse_transform_not_enough_unique_values(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``strings_from_regex``
        function using the ``instance.regex_format`` and then generates the
        ``instance.data_length`` number of data.

        Setup:
            - Initialize a ``RegexGenerator`` instance with ``enforce_uniqueness`` to ``True``.
            - Set ``data_length`` to 6.
            - Initialize a generator.

        Side Effects:
            - An error is being raised as not enough unique values can be generated.
        """
        # Setup
        instance = RegexGenerator('[A-Z]', enforce_uniqueness=True)
        instance.data_length = 6
        generator = AsciiGenerator(5)
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0
        instance.columns = ['a']
        columns_data = pd.Series()

        # Assert
        error_msg = re.escape(
            'The regex is not able to generate 6 unique values. Please use a different regex '
            "for column ('a')."
        )
        with pytest.raises(TransformerProcessingError, match=error_msg):
            instance._reverse_transform(columns_data)

    def test__reverse_transform_enforce_uniqueness_not_enough_remaining(self):
        """Test the case when ``_reverse_transform`` can't generate enough new values.

        Validate that an error is being raised stating that not enough new values can be
        generated and that the user can use ``reset_randomization`` in order to restart the
        generator.
        """
        # Setup
        instance = RegexGenerator('[A-Z]', enforce_uniqueness=True)
        instance.data_length = 6
        generator = AsciiGenerator(10)
        instance.generator = generator
        instance.generator_size = 10
        instance.generated = 9
        instance.columns = ['a']
        columns_data = pd.Series()

        # Assert
        error_msg = re.escape(
            'The regex generator is not able to generate 6 new unique values (only 1 unique '
            "value left). Please use 'reset_randomization' in order to restart the generator."
        )
        with pytest.raises(TransformerProcessingError, match=error_msg):
            instance._reverse_transform(columns_data)
