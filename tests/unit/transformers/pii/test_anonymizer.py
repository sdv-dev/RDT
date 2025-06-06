"""Test Personal Identifiable Information Transformer using Faker."""

import pickle
import re
import tempfile
from unittest.mock import Mock, call, patch

import faker
import numpy as np
import pandas as pd
import pytest

from rdt.errors import TransformerInputError, TransformerProcessingError
from rdt.transformers.categorical import LabelEncoder
from rdt.transformers.pii.anonymizer import (
    AnonymizedFaker,
    PseudoAnonymizedFaker,
)


class TestAnonymizedFaker:
    """Test class for ``AnonymizedFaker``."""

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.getattr')
    @patch('rdt.transformers.pii.anonymizer.attrgetter')
    def test_check_provider_function_baseprovider(self, mock_attrgetter, mock_getattr, mock_faker):
        """Test that ``getattr`` is being called with ``BaseProvider`` and ``function_name``.

        Mock:
            - Mock the ``getattr`` from Python to ensure that is being called with the input.
            - Mock faker and ensure that ``getattr`` is being called with ``faker.providers``.
        """
        # Setup
        mock_attrgetter.return_value = lambda x: 'module'
        mock_getattr.side_effect = ['provider', None]

        # Run
        AnonymizedFaker.check_provider_function('BaseProvider', 'function_name')

        # Assert
        assert mock_attrgetter.call_args_list[0] == call('BaseProvider')
        assert mock_getattr.call_args_list[0] == call('module', 'function_name')

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.getattr')
    @patch('rdt.transformers.pii.anonymizer.attrgetter')
    def test_check_provider_function_other_providers(
        self, mock_attrgetter, mock_getattr, mock_faker
    ):
        """Test that ``getattr`` is being called with ``provider_name`` and ``function_name``.

        Mock:
            - Mock the ``getattr`` from Python to ensure that is being called with the input.
            - Mock faker and ensure that ``getattr`` is being called with ``faker.providers``.
        """
        # Setup
        mock_attrgetter.return_value = lambda x: 'module'
        mock_getattr.side_effect = ['provider_class', None]

        # Run
        AnonymizedFaker.check_provider_function('provider_name', 'function_name')

        # Assert
        assert mock_attrgetter.call_args_list[0] == call('provider_name')
        assert mock_getattr.call_args_list[0] == call('module', 'Provider')
        assert mock_getattr.call_args_list[1] == call('provider_class', 'function_name')

    def test_check_provider_function_raise_attribute_error(self):
        """Test that ``check_provider_function`` raises an ``AttributeError``.

        Test when the provider class is not in the ``faker.providers`` an ``AttributeError`` is
        being raised or when the ``function`` is not within the ``provider``.
        """
        # Setup
        expected_message = (
            "The 'TestProvider' module does not contain a function named "
            "'TestFunction'.\nRefer to the Faker docs to find the correct function: "
            'https://faker.readthedocs.io/en/master/providers.html'
        )

        # Run
        with pytest.raises(TransformerProcessingError, match=expected_message):
            AnonymizedFaker.check_provider_function('TestProvider', 'TestFunction')

    def test__function_cardinality_rule_none(self):
        """Test that ``_function`` does not use ``faker.unique``.

        The method ``_function`` should return a call from the
        ``instance.faker.provider.<function>``.

        Mock:
            - Instance of 'AnonymizedFaker'.
            - Faker instance.
            - A function for the faker instance.

        Output:
            - Return value of mocked function.

        Side Effects:
            - The returned function, when called, has to call the `faker.<function_name>` function
              with the provided kwargs.
        """
        # setup
        instance = Mock()
        function = Mock()
        unique_function = Mock()
        function.return_value = 1

        instance.cardinality_rule = None
        instance.faker.unique.number = unique_function
        instance.faker.number = function
        instance.function_name = 'number'
        instance.function_kwargs = {'type': 'int'}

        # Run
        result = AnonymizedFaker._function(instance)

        # Assert
        unique_function.assert_not_called()
        function.assert_called_once_with(type='int')
        assert result == 1

    def test___init___enforce_uniqueness_exists(self):
        """Test `enforce_uniqueness` attribute exists."""
        # Run
        instance = AnonymizedFaker()

        # Assert
        assert instance.enforce_uniqueness is None

    def test__function_cardinality_rule_unique(self):
        """Test that ``_function`` uses the ``faker.unique``.

        The method ``_function`` should return a call from the
        ``instance.faker.unique.<function>``.

        Mock:
            - Instance of 'AnonymizedFaker'.
            - Faker instance.
            - A function for the faker instance.

        Output:
            - Return value of mocked function.

        Side Effects:
            - The returned function, when called, has to call the ``faker.unique.<function_name>``
              function with the provided kwargs.
        """
        # setup
        instance = Mock()
        function = Mock()
        unique_function = Mock()
        unique_function.return_value = 1

        instance.cardinality_rule = 'unique'
        instance.faker.unique.number = unique_function
        instance.faker.number = function
        instance.function_name = 'number'
        instance.function_kwargs = {'type': 'int'}

        # Run
        result = AnonymizedFaker._function(instance)

        # Assert
        function.assert_not_called()
        unique_function.assert_called_once_with(type='int')
        assert result == 1

    def test__function_cardinality_rule_match(self):
        """Test it when 'cardinality_rule' is 'match'."""
        # setup
        instance = Mock()
        function = Mock()
        unique_function = Mock()
        unique_function.return_value = 1

        instance.cardinality_rule = 'match'
        instance.faker.unique.number = unique_function
        instance.faker.number = function
        instance.function_name = 'number'
        instance.function_kwargs = {'type': 'int'}

        # Run
        result = AnonymizedFaker._function(instance)

        # Assert
        function.assert_not_called()
        unique_function.assert_called_once_with(type='int')
        assert result == 1

    def test__function_cardinality_rule_missing_attribute(self):
        """Test it when ``cardinality_rule`` attribute is missing."""
        # setup
        instance = Mock()
        function = Mock()
        unique_function = Mock()
        unique_function.return_value = 1

        delattr(instance, 'cardinality_rule')
        instance.enforce_uniqueness = True
        instance.faker.unique.number = unique_function
        instance.faker.number = function
        instance.function_name = 'number'
        instance.function_kwargs = {'type': 'int'}

        # Run
        result = AnonymizedFaker._function(instance)

        # Assert
        function.assert_not_called()
        unique_function.assert_called_once_with(type='int')
        assert result == 1

    def test__function_with_iterables_return(self):
        """Test that ``_function`` returns the values of the iterable."""
        # setup
        instance = Mock()
        instance.cardinality_rule = None
        function = Mock()
        function.return_value = ('value_1', 'value_2')

        instance.faker.number = function
        instance.function_name = 'number'
        instance.function_kwargs = {'type': 'int'}

        # Run
        result = AnonymizedFaker._function(instance)

        # Assert
        function.assert_called_once_with(type='int')
        assert result == 'value_1, value_2'

    @patch('rdt.transformers.pii.anonymizer.importlib')
    @patch('rdt.transformers.pii.anonymizer.warnings')
    def test__check_locales(self, mock_warnings, mock_importlib):
        """Test that check locales warns the user if the spec was not found.

        Mock:
            - Mock importlib with side effects to return one `None` and one value.
            - Mock the warnings.
        Side Effect:
            - mock_warnings has been called once with the expected message.
        """
        # Setup
        instance = Mock()
        instance.provider_name = 'credit_card'
        instance.function_name = 'credit_card_full'
        instance.locales = ['es_ES', 'en_US']
        mock_importlib.util.find_spec.side_effect = [None, 'en_US']

        # Run
        AnonymizedFaker._check_locales(instance)

        # Assert
        expected_message = (
            "Locales ['es_ES'] do not support provider 'credit_card' "
            "and function 'credit_card_full'.\nIn place of these locales, 'en_US' will "
            'be used instead. Please refer to the localized provider docs for more '
            'information: https://faker.readthedocs.io/en/master/locales.html'
        )
        mock_warnings.warn.assert_called_once_with(expected_message)

    @patch('rdt.transformers.pii.anonymizer.importlib')
    @patch('rdt.transformers.pii.anonymizer.warnings')
    def test__check_locales_provider_ending_with_locale(self, mock_warnings, mock_importlib):
        """Test that check locales does not warn the user if the provider ends with the locale.

        Mock:
            - Mock importlib with side effects to return `None`.
            - Mock the warnings.
        """
        # Setup
        instance = Mock()
        instance.provider_name = 'address.en_US'
        instance.function_name = 'postcode'
        instance.locales = ['en_US']
        mock_importlib.util.find_spec.side_effect = ['en_US']

        # Run
        AnonymizedFaker._check_locales(instance)

        # Assert
        mock_warnings.warn.assert_not_called()

    @patch('rdt.transformers.pii.anonymizer.importlib')
    @patch('rdt.transformers.pii.anonymizer.warnings')
    def test__check_locales_provider_ending_with_wrong_locale(self, mock_warnings, mock_importlib):
        """Test that check locales warns the user.

        If the provider ends with the given locale but is not separated by a dot this will warn
        that the default 'en_US' will be used instead'.

        Mock:
            - Mock importlib with side effects to return `None`.
            - Mock the warnings.
        """
        # Setup
        instance = Mock()
        instance.provider_name = 'addressit'
        instance.function_name = 'postcode'
        instance.locales = ['it']
        mock_importlib.util.find_spec.side_effect = [None]

        # Run
        AnonymizedFaker._check_locales(instance)

        # Assert
        expected_message = (
            "Locales ['it'] do not support provider 'addressit' and function 'postcode'.\n"
            "In place of these locales, 'en_US' will be used instead. "
            'Please refer to the localized provider docs for more information: '
            'https://faker.readthedocs.io/en/master/locales.html'
        )
        mock_warnings.warn.assert_called_once_with(expected_message)

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.AnonymizedFaker.check_provider_function')
    def test___init__default(self, mock_check_provider_function, mock_faker):
        """Test the default instantiation of the transformer.

        Test that by default the transformer is being instantiated with ``BaseProvider`` and
        ``lexify``.

        Mock:
            - ``check_provider_function`` mock and assert that this is being called with
              ``BaseProvider`` and ``lexify``.
            - ``faker`` mock to ensure that is being called with ``None`` as locales
              and that the ``lexify`` from the instance is being assigned to the ``_function``.

        Side effects:
            - the ``instance.provider_name`` is ``BaseProvider``.
            - the ``instance.function_name`` is ``lexify``.
            - the ``instance.locales`` is ``None``.
            - the ``instance.function_kwargs`` is an empty ``dict``.
            - ``check_provider_function`` has been called once with ``BaseProvider`` and
              ``lexify``.
            - the ``instance._function`` is ``instance.faker.lexify``.
        """
        # Run
        instance = AnonymizedFaker()

        # Assert
        mock_check_provider_function.assert_called_once_with('BaseProvider', 'lexify')
        assert instance.provider_name == 'BaseProvider'
        assert instance.function_name == 'lexify'
        assert instance.function_kwargs == {}
        assert instance.locales is None
        mock_faker.Faker.assert_called_once_with(None)
        assert instance.cardinality_rule is None
        assert instance.missing_value_generation == 'random'

    def test___init__error_missing_value_generation(self):
        """Test that the ``__init__`` raises an error if the missing value generation is invalid.

        Test that the ``__init__`` raises an error if the missing value generation is invalid.

        Side effects:
            - ``ValueError`` is raised.
        """
        expected_message = (
            "Missing value generation 'invalid' is not supported "
            "for AnonymizedFaker. Please use either 'random' or None."
        )
        # Run and Assert
        with pytest.raises(TransformerInputError, match=expected_message):
            AnonymizedFaker(missing_value_generation='invalid')

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.AnonymizedFaker.check_provider_function')
    @patch('rdt.transformers.pii.anonymizer._handle_enforce_uniqueness_and_cardinality_rule')
    def test___init__custom(self, mock__handle, mock_check_provider_function, mock_faker):
        """Test the instantiation of the transformer with custom parameters.

        Test that the transformer can be instantiated with a custom provider and function, and
        this is being assigned properly.

        Mock:
            - ``check_provider_function`` mock and assert that this is being called with
              ``CreditCard`` and ``credit_card_full``.
            - ``faker`` mock to ensure that is being called with ``['en_US', 'fr_FR']`` as locales
              and that the ``credit_card_full`` from the instance is being assigned to the
              ``_function``.

        Side effects:
            - the ``instance.provider_name`` is ``CreditCard``.
            - the ``instance.function_name`` is ``credit_card_full``.
            - the ``instance.locales`` is ``['en_US', 'fr_FR']``.
            - the ``instance.function_kwargs`` is ``{'type': 'visa'}``.
            - ``check_provider_function`` has been called once with ``CreditCard`` and
              ``credit_card_full``.
            - the ``instance._function`` is ``instance.faker.credit_card_full``.
        """
        # Setup
        mock__handle.return_value = 'unique'

        # Run
        instance = AnonymizedFaker(
            provider_name='credit_card',
            function_name='credit_card_full',
            function_kwargs={'type': 'visa'},
            locales=['en_US', 'fr_FR'],
            enforce_uniqueness=True,
        )

        # Assert
        mock_check_provider_function.assert_called_once_with('credit_card', 'credit_card_full')
        assert instance.provider_name == 'credit_card'
        assert instance.function_name == 'credit_card_full'
        assert instance.function_kwargs == {'type': 'visa'}
        assert instance.locales == ['en_US', 'fr_FR']
        mock_faker.Faker.assert_called_once_with(['en_US', 'fr_FR'])
        assert instance.cardinality_rule == 'unique'
        mock__handle.assert_called_once_with(True, None)

    def test___init__no_function_name(self):
        """Test the instantiation of the transformer with custom parameters.

        Test that the transformer raises an error when no function name is provided.

        Raises:
            - Error.
        """
        # Run / Assert
        expected_message = (
            "Please specify the function name to use from the 'credit_card' provider."
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            AnonymizedFaker(provider_name='credit_card', locales=['en_US', 'fr_FR'])

    @patch('rdt.transformers.pii.anonymizer.issubclass')
    @patch('rdt.transformers.pii.anonymizer.BaseTransformer')
    def test_get_supported_sdtypes(self, base_mock, issubclass_mock):
        """Test that the method returns all sdtypes except the basic ones."""
        # Setup
        issubclass_mock.return_value = False
        numerical_mock = Mock()
        numerical_mock.get_supported_sdtypes.return_value = ['numerical']
        categorical_mock = Mock()
        categorical_mock.get_supported_sdtypes.return_value = ['categorical']
        datetime_mock = Mock()
        datetime_mock.get_supported_sdtypes.return_value = ['datetime']
        boolean_mock = Mock()
        boolean_mock.get_supported_sdtypes.return_value = [
            'boolean',
            'categorical',
        ]
        text_mock = Mock()
        text_mock.get_supported_sdtypes.return_value = ['text']
        phone_mock = Mock()
        phone_mock.get_supported_sdtypes.return_value = ['phone_number']
        pii_mock = Mock()
        pii_mock.get_supported_sdtypes.return_value = ['pii']
        base_mock.get_subclasses.return_value = [
            numerical_mock,
            categorical_mock,
            datetime_mock,
            boolean_mock,
            text_mock,
            phone_mock,
            pii_mock,
        ]

        # Run
        supported_sdtypes = AnonymizedFaker.get_supported_sdtypes()

        # Assert
        assert sorted(supported_sdtypes) == sorted([
            'phone_number',
            'pii',
            'text',
        ])

    @patch('rdt.transformers.pii.anonymizer.BaseTransformer.reset_randomization')
    @patch('rdt.transformers.pii.anonymizer.faker')
    def test_reset_randomization(self, mock_faker, mock_base_reset):
        """Test that this function creates a new faker instance."""
        # Setup
        instance = AnonymizedFaker()
        instance.locales = ['en_US']

        # Run
        AnonymizedFaker.reset_randomization(instance)

        # Assert
        mock_faker.Faker.assert_has_calls([call(None), call(['en_US'])])
        mock_base_reset.assert_called_once()

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that the ``_fit`` method learns the size of the input data.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.

        Input:
            - ``pd.Series`` containing 3 strings.

        Side Effects:
            - ``instance.data_length`` equals to the length of the input data.
        """
        # Setup
        transformer = AnonymizedFaker()
        transformer.cardinality_rule = 'match'
        columns_data = pd.Series(['1', '2', '3', None, np.nan])
        transformer.columns = ['col']

        # Run
        transformer._fit(columns_data)

        # Assert
        assert transformer.data_length == 5
        assert transformer.output_properties == {None: {'next_transformer': None}}
        assert transformer._nan_frequency == 0.4
        assert transformer._data_cardinality == 3

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method returns ``None``.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - ``None``.
        """
        # Setup
        columns_data = pd.Series([1, 2, 3])
        instance = AnonymizedFaker()

        # Run
        result = instance._transform(columns_data)

        # Assert
        assert result is None

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``instance._function`` with
        the ``instance.function_kwargs`` the ``instance.data_length`` amount of times.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.

        Input:
            - ``None``.

        Output:
            - the output a ``numpy.array`` with the generated values from
              ``instance._function``.
        """
        # Setup
        instance = AnonymizedFaker()
        instance.data_length = 3
        function = Mock()
        function.side_effect = ['a', 'b', 'c']

        instance._function = function

        # Run
        result = instance._reverse_transform(None)

        # Assert
        assert function.call_args_list == [call(), call(), call()]
        np.testing.assert_array_equal(result, np.array(['a', 'b', 'c']))

    def test__reverse_transform_match_cardinality(self):
        """Test it calls the appropriate method."""
        # Setup
        instance = Mock()
        instance.data_length = 3
        instance.cardinality_rule = 'match'

        # Run
        AnonymizedFaker._reverse_transform(instance, None)

        # Assert
        instance._reverse_transform_cardinality_rules.assert_called_once_with(3)

    def test__reverse_transform_cardinality_rules_only_nans(self):
        """Test it with only nans."""
        # Setup
        instance = AnonymizedFaker()
        instance._nan_frequency = 1

        # Run
        result = instance._reverse_transform_cardinality_rules(3)

        # Assert
        assert pd.isna(result).all()

    def test__reverse_transform_cardinality_rules_no_missing_value(self):
        """Test it with default values."""
        # Setup
        instance = AnonymizedFaker(missing_value_generation=None)
        instance._data_cardinality = 2
        instance._nan_frequency = 0
        instance._unique_categories = ['a', 'b', 'c']
        function = Mock()
        function.side_effect = ['a', 'b', 'c']
        instance.cardinality_rule = 'match'
        instance._function = function

        # Run
        result = instance._reverse_transform_cardinality_rules(3)

        # Assert
        assert set(result) == set(['a', 'b', 'c'])

    def test__reverse_transform_cardinality_rules_scale(self):
        """Test it with scale cardinality."""
        # Setup
        instance = AnonymizedFaker(missing_value_generation=None)
        instance._data_cardinality = 2
        instance._nan_frequency = 0
        instance._data_cardinality_scale = {
            'num_repetitions': [1, 2, 3],
            'frequency': [0.1, 0.2, 0.7],
        }
        function = Mock()
        function.side_effect = ['a', 'b', 'c']
        instance.cardinality_rule = 'scale'
        instance._function = function

        # Run
        result = instance._reverse_transform_cardinality_rules(3)

        # Assert
        assert set(result).issubset(set(['a', 'b', 'c']))

    def test__reverse_transform_cardinality_rules_not_enough_unique(self):
        """Test it when there are not enough unique values."""
        # Setup
        instance = AnonymizedFaker()
        instance._data_cardinality = 3
        instance._nan_frequency = 0
        function = Mock()
        function.side_effect = ['a', 'b', 'c', 'd']
        instance._function = function
        instance.cardinality_rule = 'match'

        # Run
        result = instance._reverse_transform_cardinality_rules(6)

        # Assert
        assert set(result) == {'a', 'b', 'c'}

    def test__reverse_transform_cardinality_rule_missing_attribute(self):
        """Test that when the ``cardinality_rule`` attribute is missing."""
        # Setup
        instance = Mock()
        delattr(instance, 'cardinality_rule')

        instance.data_length = 3

        # Run
        AnonymizedFaker._reverse_transform(instance, None)

        # Assert
        instance._reverse_transform_with_fallback.assert_called_once_with(3)

    def test__reverse_transform_with_fallback_handles_uniqueness_exception(self):
        """Test that when we reverse transform with fallback we handle uniqueness."""
        # Setup
        instance = AnonymizedFaker()
        instance.get_input_column = Mock(return_value='test_column')
        instance._function = Mock(
            side_effect=['val1', 'val2', faker.exceptions.UniquenessException()]
        )
        instance._fallback_function = Mock(side_effect=['fallback1', 'fallback2', 'fallback3'])
        sample_size = 5

        # Run and Assert
        warning_msg = re.escape(
            "Unable to generate enough unique values for column 'test_column' in "
            'a human-readable format. Additional values may be created randomly.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            result = instance._reverse_transform_with_fallback(sample_size)

        # Assert
        assert instance._function.call_count == 3
        assert instance._fallback_function.call_count == 3
        assert result.tolist() == ['val1', 'val2', 'fallback1', 'fallback2', 'fallback3']

    def test__reverse_transform_with_nans(self):
        """Test that ``_reverse_transform`` generates NaNs."""
        # Setup
        instance = AnonymizedFaker()
        instance.data_length = 4
        instance._nan_frequency = 0.25
        function = Mock()
        function.side_effect = ['a', 'b', 'c', 'd']

        instance._function = function

        # Run
        result = instance._reverse_transform(None)
        result = pd.Series(result)

        # Assert
        assert function.call_args_list == [call(), call(), call(), call()]
        assert instance.missing_value_generation == 'random'
        assert result.isna().sum() == 1

    def test__reverse_transform_not_enough_unique_values(self):
        """Test the ``_reverse_transform`` method.

        Test that when calling the ``_reverse_transform`` method and the ``instance._function`` is
        not generating enough unique values raises an error.

        Setup:
            -Instance of ``AnonymizedFaker``.

        Input:
            - ``pandas.Series`` representing a column.

        Side Effect:
            - Raises an error.
        """
        # Setup
        instance = AnonymizedFaker('misc', 'boolean', cardinality_rule='unique')
        data = pd.Series(['a', 'b', 'c', 'd'])
        instance.columns = ['a']

        # Run / Assert
        error_msg = re.escape(
            "Unable to generate enough unique values for column 'a' in "
            'a human-readable format. Additional values may be created randomly.'
        )
        with pytest.warns(UserWarning, match=error_msg):
            instance._reverse_transform(data)

    def test__reverse_transform_size_is_length_of_data(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``instance._function`` with
        the ``instance.function_kwargs`` the ``len(data)`` amount of times.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - the output a ``numpy.array`` with the generated values from
              ``instance._function``.
        """
        # Setup
        instance = AnonymizedFaker()
        data = pd.Series([1, 2, 3])
        instance.data_length = 0
        function = Mock()
        function.side_effect = ['a', 'b', 'c']

        instance._function = function

        # Run
        result = instance._reverse_transform(data)

        # Assert
        assert function.call_args_list == [call(), call(), call()]
        np.testing.assert_array_equal(result, np.array(['a', 'b', 'c']))

    def test__reverse_transform_scale(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 50 + ['B'] * 100})
        instance = AnonymizedFaker(cardinality_rule='scale')
        instance.fit(data, 'col')

        # Run
        out = instance._reverse_transform(data)

        # Assert
        assert out[out == 'KAab'].size in {50, 100, 150}
        assert out[out == 'qOSU'].size in {0, 50, 100}

    def test__reverse_transform_scale_multiple_calls(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 50 + ['B'] * 50 + ['C'] * 50})
        instance = AnonymizedFaker(cardinality_rule='scale')
        instance.fit(data, 'col')

        # Run
        out1 = instance._reverse_transform(data)
        out2 = instance._reverse_transform(data)
        instance.reset_randomization()
        out3 = instance._reverse_transform(data)

        # Assert
        assert out1[out1 == 'KAab'].size == 50
        assert out1[out1 == 'qOSU'].size == 50
        assert out1[out1 == 'CPmg'].size == 50

        assert out2[out2 == 'urbw'].size == 50
        assert out2[out2 == 'JEWW'].size == 50
        assert out2[out2 == 'LRyt'].size == 50

        assert out3[out3 == 'KAab'].size == 50
        assert out3[out3 == 'qOSU'].size == 50
        assert out1[out1 == 'CPmg'].size == 50

    def test__reverse_transform_scale_remaining_values(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 10 + ['B'] * 3})
        instance = AnonymizedFaker(cardinality_rule='scale')
        instance.fit(data, 'col')

        # Run
        out1 = instance._reverse_transform(data.head(8))
        out2 = instance._reverse_transform(data)

        # Assert
        assert out1[out1 == 'qOSU'].size == 3
        assert out1[out1 == 'KAab'].size == 5
        assert out2[out2 == 'KAab'].size == 5
        assert out2[out2 == 'CPmg'].size == 8

    def test__reverse_transform_scale_many_remaining_values(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 100})
        instance = AnonymizedFaker(cardinality_rule='scale')
        instance.fit(data, 'col')

        # Run
        out1 = instance._reverse_transform(data.head(10))
        out2 = instance._reverse_transform(data.head(10))

        # Assert
        assert np.array_equal(out1, np.array(['qOSU'] * 10))
        assert np.array_equal(out2, np.array(['qOSU'] * 10))

    def test__set_fitted_parameters(self):
        """Test ``_set_fitted_parameters`` sets the required parameters for transformer."""
        # Setup
        transformer = AnonymizedFaker()
        frequency = 0.30
        cardinality = 3
        column_name = 'mock'

        # Run
        transformer.cardinality_rule = 'match'
        error_msg = re.escape('Cardinality "match" rule must specify a cardinality value.')
        with pytest.raises(TransformerInputError, match=error_msg):
            transformer._set_fitted_parameters(column_name, nan_frequency=frequency)

        transformer.cardinality_rule = 'scale'
        error_msg = re.escape('Cardinality "scale" rule must specify a cardinality value.')
        with pytest.raises(TransformerInputError, match=error_msg):
            transformer._set_fitted_parameters(column_name, nan_frequency=frequency)

        transformer._set_fitted_parameters(
            column_name,
            nan_frequency=frequency,
            cardinality=cardinality,
            cardinality_scale={'num_repetitions': [1, 2, 3], 'frequency': [0.1, 0.2, 0.7]},
        )

        # Assert
        assert transformer._nan_frequency == frequency
        assert transformer._data_cardinality == cardinality
        assert transformer.columns == [column_name]
        assert transformer._data_cardinality_scale == {
            'num_repetitions': [1, 2, 3],
            'frequency': [0.1, 0.2, 0.7],
        }

    def test___repr__default(self):
        """Test the ``__repr__`` method.

        With the default parameters should return only the ``function_name='lexify'`` as an
        starting argument for the ``AnonymizedFaker``.
        """
        # Setup
        instance = AnonymizedFaker()

        # Run
        res = repr(instance)

        # Assert
        expected_res = "AnonymizedFaker(function_name='lexify')"
        assert res == expected_res

    def test___repr__custom_provider(self):
        """Test the ``__repr__`` method.

        With the custom args the ``repr`` of the class should return all the non default
        arguments.
        """
        # Setup
        instance = AnonymizedFaker('credit_card', 'credit_card_full')

        # Run
        res = repr(instance)

        # Assert
        expected = "AnonymizedFaker(provider_name='credit_card', function_name='credit_card_full')"
        assert res == expected


class TestPseudoAnonymizedFaker:
    """Test class for ``PseudoAnonymizedFaker``."""

    @patch('rdt.transformers.pii.anonymizer.warnings')
    def test___getstate__(self, mock_warnings):
        """Test that when pickling the warning message is being triggered."""
        # Setup
        instance = PseudoAnonymizedFaker()

        expected_warning_msg = (
            'You are saving the mapping information, which includes the original data. '
            'Sharing this object with others will also give them access to the original data '
            'used with this transformer.'
        )

        # Run / Assert
        with tempfile.TemporaryFile() as tmp:
            pickle.dump(instance, tmp)

        mock_warnings.warn.assert_called_once_with(expected_warning_msg)

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.AnonymizedFaker.check_provider_function')
    def test___init__super_attrs(self, mock_check_provider_function, mock_faker):
        """Test that initializing an instance is calling properly the ``super`` class.

        Mock:
            - ``check_provider_function`` mock and assert that this is being called with
              ``BaseProvider`` and ``lexify``.
            - ``faker`` mock to ensure that is being called with ``None`` as locales
              and that the ``lexify`` from the instance is being assigned to the ``_function``.

        Side effects:
            - the ``instance.provider_name`` is ``BaseProvider``.
            - the ``instance.function_name`` is ``lexify``.
            - the ``instance.locales`` is ``None``.
            - the ``instance.function_kwargs`` is an empty ``dict``.
            - ``check_provider_function`` has been called once with ``BaseProvider`` and
              ``lexify``.
            - the ``instance._function`` is ``instance.faker.lexify``.
        """
        # Run
        instance = PseudoAnonymizedFaker()

        # Assert
        assert instance._mapping_dict == {}
        assert instance._reverse_mapping_dict == {}

        # Assert Super Attrs
        assert instance.provider_name == 'BaseProvider'
        assert instance.function_name == 'lexify'
        assert instance.function_kwargs == {}
        assert instance.locales is None
        mock_faker.Faker.assert_called_once_with(None)

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.AnonymizedFaker.check_provider_function')
    def test___init__custom(self, mock_check_provider_function, mock_faker):
        """Test the instantiation of the transformer with custom parameters.

        Test that the transformer can be instantiated with a custom provider and function, and
        this is being assigned properly.

        Mock:
            - ``check_provider_function`` mock and assert that this is being called with
              ``CreditCard`` and ``credit_card_full``.
            - ``faker`` mock to ensure that is being called with ``['en_US', 'fr_FR']`` as locales
              and that the ``credit_card_full`` from the instance is being assigned to the
              ``_function``.

        Side effects:
            - the ``instance.provider_name`` is ``CreditCard``.
            - the ``instance.function_name`` is ``credit_card_full``.
            - the ``instance.locales`` is ``['en_US', 'fr_FR']``.
            - the ``instance.function_kwargs`` is ``{'type': 'visa'}``.
            - ``check_provider_function`` has been called once with ``CreditCard`` and
              ``credit_card_full``.
            - the ``instance._function`` is ``instance.faker.credit_card_full``.
        """
        # Run
        instance = PseudoAnonymizedFaker(
            provider_name='credit_card',
            function_name='credit_card_full',
            function_kwargs={'type': 'visa'},
            locales=['en_US', 'fr_FR'],
        )

        # Assert
        assert instance._mapping_dict == {}
        assert instance._reverse_mapping_dict == {}
        mock_check_provider_function.assert_called_once_with('credit_card', 'credit_card_full')
        assert instance.provider_name == 'credit_card'
        assert instance.function_name == 'credit_card_full'
        assert instance.function_kwargs == {'type': 'visa'}
        assert instance.locales == ['en_US', 'fr_FR']
        mock_faker.Faker.assert_called_once_with(['en_US', 'fr_FR'])

    def test_get_mapping(self):
        """Test the ``get_mapping`` method.

        Validate that the ``get_mapping`` method return a ``deepcopy``  of the
        ``instance._mapping_dict``.

        Setup:
            - Instance of ``PseudoAnonymizedFaker``.
            - ``instance._mapping_dict`` to contain a dictionary.

        Output:
            - A deepcopy of the ``instance._mapping_dict``.
        """
        # Setup
        instance = PseudoAnonymizedFaker()
        instance._mapping_dict = {'a': 'b'}

        # Run
        result = instance.get_mapping()

        # Assert
        assert result == {'a': 'b'}
        assert id(result) != id(instance._mapping_dict)

    def test__fit(self):
        """Test the ``_fit`` method.

        Test that when calling the ``_fit`` method we are populating the ``instance._mapping_dict``
        with unique values and the ``instance._reverse_mapping_dict`` with those as well.

        Setup:
            -Instance of ``PseudoAnonymizedFaker``.

        Input:
            - ``pandas.Series`` representing a column.

        Mock:
            - Mock the ``instance._function`` to return controlled values.

        Side Effects:
            - ``instance._mapping_dict`` has been populated with the input unique data as keys and
              ``_function`` returned values as values.
            - ``instance._reverse_mapping_dict`` contains the ``_function`` returned values as keys
              and the input data as values.
        """
        # Setup
        instance = PseudoAnonymizedFaker()
        instance._function = Mock()
        instance._function.side_effect = [1, 2, 3]
        instance.columns = ['col']
        data = pd.Series(['a', 'b', 'c'])

        # Run
        instance._fit(data)

        # Assert
        assert instance._mapping_dict == {'a': 1, 'b': 2, 'c': 3}
        assert instance._reverse_mapping_dict == {1: 'a', 2: 'b', 3: 'c'}
        assert list(instance.output_properties) == [None]
        assert list(instance.output_properties[None]) == [
            'sdtype',
            'next_transformer',
        ]
        assert instance.output_properties[None]['sdtype'] == 'categorical'

        transformer = instance.output_properties[None]['next_transformer']
        assert isinstance(transformer, LabelEncoder)
        assert transformer.add_noise is True

    def test__fit_not_enough_unique_values_in_faker_function(self):
        """Test the ``_fit`` method.

        Test that when calling the ``_fit`` method and the ``instance._function`` is not
        generating enough unique values uses the fallback function.

        Setup:
            -Instance of ``PseudoAnonymizedFaker``.

        Input:
            - ``pandas.Series`` representing a column.

        Mock:
            - Mock the ``instance._function`` to return only 1 value.

        Side Effect:
            - Raises an error.
        """
        # Setup
        instance = PseudoAnonymizedFaker('misc', 'boolean')
        instance.columns = ['col']
        data = pd.Series(['a', 'b', 'c', 'd'])

        # Run / Assert
        warning_msg = re.escape(
            "Unable to generate enough unique values for column 'col' in "
            'a human-readable format. Additional values may be created randomly.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            instance._fit(data)

    def test__transform(self):
        """Test the ``_transform`` method.

        Test that when ``_transform`` is being performed and no new values are present, the data
        is being mapped to the ``instance._mapping_dict``.

        Setup:
            - Instance of ``PseudoAnonymizedTransformer``.
            - ``_mapping_dict`` to the original data.

        Input:
            - pandas.Series with some data to be mapped.

        Output:
            - The ``pandas.Series`` with the ``mapped_data``.
        """
        # Setup
        instance = PseudoAnonymizedFaker()
        instance._mapping_dict = {'a': 'z', 'b': 'y', 'c': 'x'}
        instance.columns = ['col']

        data = pd.Series(['a', 'b', 'c'], name='col')

        # Run
        result = instance._transform(data)

        # Assert
        pd.testing.assert_series_equal(result, pd.Series(['z', 'y', 'x'], name='col'))

    def test__transform_with_new_values(self):
        """Test the ``_transform`` method.

        Test that when new data is being passed to the ``_transform``, this raises an error.

        Setup:
            - Instance of ``PseudoAnonymizedTransformer``.
            - ``_mapping_dict`` to the original data.

        Input:
            - pandas.Series with values that are not within the ``_mapping_dict``.

        Side Effects:
            - Raises a error.
        """
        # Setup
        instance = PseudoAnonymizedFaker()
        instance._mapping_dict = {'a': 'z', 'b': 'y', 'c': 'x'}

        # Assert / Run
        error_msg_short = re.escape(
            'The data you are transforming has new, unexpected values '
            '(1, 2, 3). Please fit the transformer again using this '
            'new data.'
        )
        error_msg_long = re.escape(
            'The data you are transforming has new, unexpected values '
            '(1, 2, 3, 4, 5 and 5 more). Please fit the transformer again using this '
            'new data.'
        )

        with pytest.raises(TransformerProcessingError, match=error_msg_short):
            instance._transform(pd.Series([1, 2, 3]))

        with pytest.raises(TransformerProcessingError, match=error_msg_long):
            instance._transform(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method.

        Test that the ``_reverse_transform`` returns the input data.

        Setup:
            - instance of ``PseudoAnonymizedFaker``.

        Input:
            - pd.Series

        Output:
            - The input data.
        """
        # Setup
        instance = PseudoAnonymizedFaker()
        instance.columns = ['col']

        data = pd.Series(['a', 'b', 'c'], name='col')

        # Run
        reverse_transformed = instance._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(reverse_transformed, pd.Series(['a', 'b', 'c'], name='col'))
