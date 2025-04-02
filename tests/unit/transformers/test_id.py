"""Test for ID transformers."""

import re
from string import ascii_uppercase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.id import IDGenerator, RegexGenerator


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
        assert result.tolist() == [
            'prefix_100_suffix',
            'prefix_101_suffix',
            'prefix_102_suffix',
        ]
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
            'cardinality_rule': None,
            'generated': 0,
            'generator_size': 380204032,
            'output_properties': {None: {'next_transformer': None}},
            'regex_format': '[A-Za-z]{5}',
            'random_states': mock_random_sates,
            'generation_order': 'alphanumeric',
            '_unique_regex_values': None,
            '_data_cardinality': None,
        }

    @patch('rdt.transformers.id.strings_from_regex')
    def test___setstate__generated_and_generator_size(self, mock_strings_from_regex):
        """Test that ``__setstate__`` will initialize a generator and wind it forward."""
        # Setup
        state = {
            'data_length': None,
            'cardinality_rule': None,
            'generated': 10,
            'generator_size': 380204032,
            'output_properties': {None: {'next_transformer': None}},
            'regex_format': '[A-Za-z]{5}',
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

    @patch('rdt.transformers.id.strings_from_regex')
    def test___setstate__(self, mock_strings_from_regex):
        """Test that ``__setstate__`` will initialize a generator but not forward it.

        When ``generated`` is ``None`` and ``generator_size`` is ``None`` this will be assigned
        the ``0`` and the ``generator_size`` respectively.
        """
        # Setup
        state = {
            'data_length': None,
            'cardinality_rule': None,
            'generated': None,
            'generator_size': None,
            'output_properties': {None: {'next_transformer': None}},
            'regex_format': '[A-Za-z]{5}',
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
        assert instance.cardinality_rule is None
        assert instance.generation_order == 'alphanumeric'

    def test___init__custom(self):
        """Test __init__ with custom parameters."""
        # Run
        instance = RegexGenerator(
            regex_format='[0-9]',
            cardinality_rule='unique',
            generation_order='scrambled',
        )

        # Assert
        assert instance.data_length is None
        assert instance.regex_format == '[0-9]'
        assert instance.cardinality_rule == 'unique'
        assert instance.generation_order == 'scrambled'

    def test___init__cardinality_rule_match(self):
        """Test it when cardinality_rule is 'match'."""
        # Run
        instance = RegexGenerator(
            regex_format='[0-9]',
            cardinality_rule='match',
        )

        # Assert
        assert instance.data_length is None
        assert instance.regex_format == '[0-9]'
        assert instance.cardinality_rule == 'match'
        assert instance._data_cardinality is None
        assert instance._unique_regex_values is None

    def test___init__bad_value_generation_order(self):
        """Test that an error is raised if a bad value is given for `generation_order`."""
        # Run and Assert
        error_message = "generation_order must be one of 'alphanumeric' or 'scrambled'."
        with pytest.raises(ValueError, match=error_message):
            RegexGenerator(generation_order='afdsfd')

    def test__init__with_enforce_uniqueness(self):
        """Test that the ``enforce_uniqueness`` parameter is deprecated."""
        # Setup
        expected_message = re.escape(
            "The 'enforce_uniqueness' parameter is no longer supported. "
            "Please use the 'cardinality_rule' parameter instead."
        )

        # Run
        with pytest.warns(FutureWarning, match=expected_message):
            instance_1 = RegexGenerator(enforce_uniqueness=True, cardinality_rule='unique')

        with pytest.warns(FutureWarning, match=expected_message):
            RegexGenerator('A-Za-z', None, 'alphanumeric', True)

        with pytest.warns(FutureWarning, match=expected_message):
            instance_2 = RegexGenerator(enforce_uniqueness=True)

        with pytest.warns(FutureWarning, match=expected_message):
            instance_3 = RegexGenerator(enforce_uniqueness=False)

        # Assert
        assert instance_1.cardinality_rule == 'unique'
        assert instance_2.cardinality_rule == 'unique'
        assert instance_3.cardinality_rule is None

    @patch('rdt.transformers.id.BaseTransformer.reset_randomization')
    @patch('rdt.transformers.id.strings_from_regex')
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

    def test__fit_cardinality_rule_match(self):
        """Test it when cardinality_rule is 'match'."""
        # Setup
        instance = RegexGenerator(cardinality_rule='match')
        columns_data = pd.Series(['1', '2', '3', '2', '1'])

        # Run
        instance._fit(columns_data)

        # Assert
        assert instance.data_length == 5
        assert instance._data_cardinality == 3
        assert instance._unique_regex_values == ['AAAAA', 'AAAAB', 'AAAAC']
        assert instance.output_properties == {None: {'next_transformer': None}}

    def test__fit_cardinality_rule_match_with_regex_format(self):
        """Test it when cardinality_rule is 'match'."""
        # Setup
        instance = RegexGenerator(cardinality_rule='match', regex_format='[1-5]{1}')
        columns_data = pd.Series(['1', '2', '3', '2', '1'])

        # Run
        instance._fit(columns_data)

        # Assert
        assert instance.data_length == 5
        assert instance._data_cardinality == 3
        assert instance._unique_regex_values == ['1', '2', '3']
        assert instance.output_properties == {None: {'next_transformer': None}}

    def test__fit_cardinality_rule_match_with_nans(self):
        """Test it when cardinality_rule is 'match'."""
        # Setup
        instance = RegexGenerator(cardinality_rule='match', regex_format='[1-5]{1}')
        columns_data = pd.Series(['1', '2', '3', '2', '1', np.nan, None, np.nan])

        # Run
        instance._fit(columns_data)

        # Assert
        assert instance.data_length == 8
        assert instance._data_cardinality == 4
        assert instance._unique_regex_values == ['1', '2', '3', '4']
        assert instance.output_properties == {None: {'next_transformer': None}}

    def test__fit_cardinality_rule_match_with_nans_too_many_values(self):
        """Test it when cardinality_rule is 'match'."""
        # Setup
        instance = RegexGenerator(cardinality_rule='match', regex_format='[1-3]{1}')
        columns_data = pd.Series(['1', '2', '3', '2', '4'])

        # Run
        instance._fit(columns_data)

        # Assert
        assert instance.data_length == 5
        assert instance._data_cardinality == 4
        assert instance._unique_regex_values == ['1', '2', '3', '4']
        assert instance.output_properties == {None: {'next_transformer': None}}

    def test__fit_cardinality_rule_match_with_too_many_values_str(self):
        """Test it when cardinality_rule is 'match'."""
        # Setup
        instance = RegexGenerator(cardinality_rule='match', regex_format='[a-b]{1}')
        columns_data = pd.Series(['a', 'b', 'c', 'b', 'd', 'f'])

        # Run
        instance._fit(columns_data)

        # Assert
        assert instance.data_length == 6
        assert instance._data_cardinality == 5
        assert instance._unique_regex_values == ['a', 'b', 'a(0)', 'b(0)', 'a(1)']
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

    @patch('rdt.transformers.id.np.random.shuffle')
    def test__reverse_transform_generation_order_scrambled(self, shuffle_mock):
        """Test the ``_reverse_transform`` method with ``generation_order`` set to scrambled.

        Validate that when ``generation_order`` is ``'scrambled'``, the data is not in order.
        """
        # Setup
        instance = RegexGenerator('[A-Z]')
        columns_data = pd.Series()
        instance.data_length = 3
        generator = AsciiGenerator(max_size=5)
        instance.generator = generator
        instance.generator_size = 5
        instance.generated = 0
        instance.generation_order = 'scrambled'
        instance.columns = ['col']

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        np.testing.assert_array_equal(result, np.array(['A', 'B', 'C']))
        shuffle_mock.assert_called_once_with(['A', 'B', 'C'])

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
        instance.columns = ['col']

        # Run
        result = instance._reverse_transform(columns_data)

        # Assert
        np.testing.assert_array_equal(result, np.array(['A', 'B', 'C']))

    def test__reverse_transform_generator_size_smaller_than_data_length(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method uses the ``instance.generator``
        to generate the ``instance.data_length`` number of data when ``cardinality_rule`` is
        ``unique`` but the data to be created is bigger.

        Setup:
            - Initialize a ``RegexGenerator`` instance.
            - Set ``data_length`` to 11.
            - Initialize a generator.

        Output:
            - A ``numpy.array`` with the first five letters from the generator repeated.
        """
        # Setup
        instance = RegexGenerator('[A-Z]', cardinality_rule=None)
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
        expected_result = np.array([
            'A',
            'B',
            'C',
            'D',
            'E',
            'A',
            'B',
            'C',
            'D',
            'E',
            'A',
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test__reverse_transform_generator_size_of_input_data(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method uses the ``instance.generator``
        to generate the ``instance.data_length`` number of data when ``cardinality_rule`` is
        ``None`` but the data to be created is bigger.

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

    @patch('rdt.transformers.id.warnings')
    def test__reverse_transform_not_enough_unique_values_cardniality_rule(self, mock_warnings):
        """Test it when there are not enough unique values to generate."""
        # Setup
        instance = RegexGenerator('[A-E]', cardinality_rule='unique')
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
        instance = RegexGenerator('[A-E]', cardinality_rule=None)
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

    @patch('rdt.transformers.id.warnings')
    def test__reverse_transform_not_enough_unique_values_numerical(self, mock_warnings):
        """Test it when there are not enough unique values to generate."""
        # Setup
        instance = RegexGenerator('[1-3]', cardinality_rule='unique')
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

    @patch('rdt.transformers.id.warnings')
    def test__reverse_transform_unique_not_enough_remaining(self, mock_warnings):
        """Test the case when there are not enough unique values remaining."""
        # Setup
        instance = RegexGenerator('[A-Z]', cardinality_rule='unique')
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

    @patch('rdt.transformers.id.LOGGER')
    def test__reverse_transform_info_message(self, mock_logger):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method logs an info message when
        ``enforce_uniqueness`` is ``False`` and the ``instance.data_length`` is bigger than
        ``instance.generator_size``.

        In this test we also test the backward compatibility, so when the transformer
        does not have the ``cardinality_rule`` attribute, it should use the ``enforce_uniqueness``
        attribute. This is necessary to keep a coverage of 100%.
        """
        # Setup
        instance = RegexGenerator('[A-Z]', cardinality_rule=None)
        del instance.cardinality_rule
        instance.enforce_uniqueness = False
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

    def test__reverse_transform_match_not_enough_values(self):
        """Test the case when there are not enough values to match the cardinality rule."""
        # Setup
        data = pd.DataFrame({'col': ['A', 'B', 'C', 'D', 'E']})
        instance = RegexGenerator(regex_format='[A-Z]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run and Assert
        warn_msg = re.escape(
            'Only 3 values can be generated. Cannot match the cardinality '
            'of the data, it requires 5 values.'
        )
        with pytest.warns(UserWarning, match=warn_msg):
            out = instance._reverse_transform(data[:3])

        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C']))

    def test__reverse_transform_match_too_many_samples(self):
        """Test it when the number of samples is bigger than the generator size."""
        # Setup
        data = pd.DataFrame({'col': ['A', 'B', 'C', 'D', 'E']})
        instance = RegexGenerator(regex_format='[A-C]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run and Assert
        warn_msg = re.escape(
            "The regex for 'col' can only generate 3 unique values. Additional values may not "
            'exactly follow the provided regex.'
        )
        with pytest.warns(UserWarning, match=warn_msg):
            out = instance._reverse_transform(data)

        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'A(0)', 'B(0)']))

    def test__reverse_transform_only_one_warning(self):
        """Test it when the num_samples < generator_size but data_cardinality > generator_size."""
        # Setup
        data = pd.DataFrame({'col': ['A', 'B', 'C', 'D', 'E']})
        instance = RegexGenerator(regex_format='[A-C]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run and Assert
        warn_msg = re.escape(
            'Only 2 values can be generated. Cannot match the cardinality of the data, '
            'it requires 5 values.'
        )
        with pytest.warns(UserWarning, match=warn_msg):
            out = instance._reverse_transform(data[:2])

        np.testing.assert_array_equal(out, np.array(['A', 'B']))
        instance._unique_regex_values = ['A', 'B', 'C', 'A(0)', 'B(0)']

    def test__reverse_transform_two_warnings(self):
        """Test it when data_cardinality > num_samples > generator_size."""
        # Setup
        data = pd.DataFrame({'col': ['A', 'B', 'C', 'D', 'E']})
        instance = RegexGenerator(regex_format='[A-C]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run and Assert
        warn_msg_1 = re.escape(
            'Only 4 values can be generated. Cannot match the cardinality of the data, '
            'it requires 5 values.'
        )
        warn_msg_2 = re.escape(
            "The regex for 'col' can only generate 3 unique values. Additional values may "
            'not exactly follow the provided regex.'
        )
        with pytest.warns(UserWarning, match=warn_msg_1):
            with pytest.warns(UserWarning, match=warn_msg_2):
                out = instance._reverse_transform(data[:4])

        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'A(0)']))

    def test__reverse_transform_match_empty_data(self):
        """Test it when the data is empty and the cardinality rule is 'match'."""
        # Setup
        data = pd.DataFrame({'col': ['A', 'B', 'C', 'D', 'E']})
        instance = RegexGenerator(regex_format='[A-Z]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run
        out = instance._reverse_transform(pd.Series())

        # Assert
        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'D', 'E']))

    def test__reverse_transform_match_with_nans(self):
        """Test it when the data has nans and the cardinality rule is 'match'."""
        # Setup
        data = pd.DataFrame({'col': ['A', np.nan, np.nan, 'D', np.nan]})
        instance = RegexGenerator(regex_format='[A-Z]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run
        out = instance._reverse_transform(data)

        # Assert
        np.testing.assert_array_equal(out, np.array(['A', 'B', 'C', 'A', 'B']))

    def test__reverse_transform_match_too_many_values(self):
        """Test it when the data has more values than the cardinality rule."""
        # Setup
        data = pd.DataFrame({'col': ['A', 'B', 'B', 'C']})
        instance = RegexGenerator(regex_format='[A-Z]', cardinality_rule='match')
        instance.fit(data, 'col')

        # Run
        out = instance._reverse_transform(pd.Series([1] * 10))

        # Assert
        np.testing.assert_array_equal(
            out, np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'])
        )

    def test__reverse_transform_no_unique_regex_values_attribute(self):
        """Test it without the _unique_regex_values attribute."""
        # Setup
        instance = RegexGenerator('[A-E]')
        delattr(instance, '_unique_regex_values')
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
