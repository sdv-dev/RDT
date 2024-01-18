"""Test Text Transformers."""

from string import ascii_uppercase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers.text import IDGenerator, RegexGenerator


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


class TestIDGenerator:

    def test___init__default(self):
        """Test the ``__init__`` method."""
        # Run
        transformer = IDGenerator()

        # Assert
        assert transformer.prefix is None
        assert transformer.starting_value == 0
        assert transformer.suffix is None
        assert transformer._counter == 0
        assert transformer.output_properties == {None: {'next_transformer': None}}

    def test___init__with_parameters(self):
        """Test the ``__init__`` method with paremeters."""
        # Run
        transformer_prefix = IDGenerator(prefix='prefix_')
        transformer_suffix = IDGenerator(suffix='_suffix')
        transformer_starting_value = IDGenerator(starting_value=10)
        transformer_all = IDGenerator(prefix='prefix_', starting_value=10, suffix='_suffix')

        # Assert
        assert transformer_prefix.prefix == 'prefix_'
        assert transformer_prefix.starting_value == 0
        assert transformer_prefix.suffix is None
        assert transformer_prefix._counter == 0
        assert transformer_prefix.output_properties == {None: {'next_transformer': None}}

        assert transformer_suffix.prefix is None
        assert transformer_suffix.starting_value == 0
        assert transformer_suffix.suffix == '_suffix'
        assert transformer_suffix._counter == 0
        assert transformer_suffix.output_properties == {None: {'next_transformer': None}}

        assert transformer_starting_value.prefix is None
        assert transformer_starting_value.starting_value == 10
        assert transformer_starting_value.suffix is None
        assert transformer_starting_value._counter == 0
        assert transformer_starting_value.output_properties == {None: {'next_transformer': None}}

        assert transformer_all.prefix == 'prefix_'
        assert transformer_all.starting_value == 10
        assert transformer_all.suffix == '_suffix'
        assert transformer_all._counter == 0
        assert transformer_all.output_properties == {None: {'next_transformer': None}}

    def test_reset_randomization(self):
        """Test the ``reset_randomization`` method."""
        # Setup
        transformer = IDGenerator()
        transformer._counter = 10

        # Run
        transformer.reset_randomization()

        # Assert
        assert transformer._counter == 0

    def test__fit(self):
        """Test the ``_fit`` method."""
        # Setup
        transformer = IDGenerator()

        # Run
        transformer._fit(None)

        # Assert
        assert True

    def test__transform(self):
        """Test the ``_transform`` method."""
        # Setup
        transformer = IDGenerator()

        # Run
        result = transformer._transform(None)

        # Assert
        assert result is None

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method."""
        # Setup
        transformer = IDGenerator()
        transformer._counter = 10

        # Run
        result = transformer._reverse_transform(np.array([1, 2, 3]))

        # Assert
        assert isinstance(result, pd.Series)
        assert result.tolist() == ['10', '11', '12']
        assert transformer._counter == 13

    def test__reverse_transform_with_everything(self):
        """Test the ``_reverse_transform`` method with all parameters."""
        # Setup
        transformer = IDGenerator(prefix='prefix_', starting_value=100, suffix='_suffix')

        # Run
        result = transformer._reverse_transform(np.array([1, 2, 3]))

        # Assert
        assert isinstance(result, pd.Series)
        assert result.tolist() == ['prefix_100_suffix', 'prefix_101_suffix', 'prefix_102_suffix']
        assert transformer._counter == 3


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

    @patch('rdt.transformers.text.warnings')
    def test__reverse_transform_not_enough_unique_values_enforce_uniqueness(self, mock_warnings):
        """Test it when there are not enough unique values to generate."""
        # Setup
        instance = RegexGenerator('[A-E]', enforce_uniqueness=True)
        instance.data_length = 6
        generator = AsciiGenerator(5)
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0
        instance.columns = ['a']
        columns_data = pd.Series()

        # Run
        out = instance._reverse_transform(columns_data)

        # Assert
        mock_warnings.warn.assert_called_once_with(
            "The regex for 'a' can only generate 5 "
            'unique values. Additional values may not exactly follow the provided regex.'
        )
        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'D', 'E', 'A(0)']))

    def test__reverse_transform_not_enough_unique_values(self):
        """Test it when there are not enough unique values to generate."""
        # Setup
        instance = RegexGenerator('[A-E]', enforce_uniqueness=False)
        instance.data_length = 6
        generator = AsciiGenerator(5)
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0
        instance.columns = ['a']
        columns_data = pd.Series()

        # Run
        out = instance._reverse_transform(columns_data)

        # Assert
        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'D', 'E', 'A']))

    @patch('rdt.transformers.text.warnings')
    def test__reverse_transform_not_enough_unique_values_numerical(self, mock_warnings):
        """Test it when there are not enough unique values to generate."""
        # Setup
        instance = RegexGenerator('[1-3]', enforce_uniqueness=True)
        instance.data_length = 6
        generator = AsciiGenerator(5)
        instance.generator = generator
        instance.generator_size = 3
        instance.generated = 0
        instance.columns = ['a']
        columns_data = pd.Series()

        # Run
        out = instance._reverse_transform(columns_data)

        # Assert
        mock_warnings.warn.assert_called_once_with(
            "The regex for 'a' can only generate 3 "
            'unique values. Additional values may not exactly follow the provided regex.'
        )
        np.testing.assert_array_equal(out, np.array(['1', '2', '3', '4', '5', '6']))

    @patch('rdt.transformers.text.warnings')
    def test__reverse_transform_enforce_uniqueness_not_enough_remaining(self, mock_warnings):
        """Test the case when there are not enough unique values remaining."""
        # Setup
        instance = RegexGenerator('[A-Z]', enforce_uniqueness=True)
        instance.data_length = 6
        generator = AsciiGenerator(10)
        instance.generator = generator
        instance.generator_size = 10
        instance.generated = 9
        instance.columns = ['a']
        columns_data = pd.Series()

        # Run
        out = instance._reverse_transform(columns_data)

        # Assert
        mock_warnings.warn.assert_called_once_with(
            'The regex generator is not able to generate 6 new unique '
            'values (only 1 unique values left).'
        )
        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'D', 'E', 'F']))

    @patch('rdt.transformers.text.LOGGER')
    def test__reverse_transform_info_message(self, mock_logger):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method logs an info message when
        ``enforce_uniqueness`` is ``False`` and the ``instance.data_length`` is bigger than
        ``instance.generator_size``.
        """
        # Setup
        instance = RegexGenerator('[A-Z]', enforce_uniqueness=False)
        instance.data_length = 6
        instance.generator_size = 5
        instance.generated = 0
        instance.columns = ['a']
        columns_data = pd.Series()

        # Run
        instance._reverse_transform(columns_data)

        # Assert
        expected_format = (
            "The data has %s rows but the regex for '%s' can only create %s unique values. Some "
            "values in '%s' may be repeated."
        )
        expected_args = (6, 'a', 5, 'a')

        mock_logger.info.assert_called_once_with(expected_format, *expected_args)
