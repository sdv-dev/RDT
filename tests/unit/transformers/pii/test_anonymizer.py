"""Test Personal Identifiable Information Transformer using Faker."""

from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from rdt.errors import Error
from rdt.transformers.null import NullTransformer
from rdt.transformers.pii.anonymizer import AnonymizedFaker


class TestAnonymizedFaker:
    """Test class for ``AnonymizedFaker``."""

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.getattr')
    def test_check_provider_function_baseprovider(self, mock_getattr, mock_faker):
        """Test that ``getattr`` is being called with ``BaseProvider`` and ``function_name``.

        Mock:
            - Mock the ``getattr`` from Python to ensure that is being called with the input.
            - Mock faker and ensure that ``getattr`` is being called with ``faker.providers``.
        """
        # Setup
        mock_getattr.side_effect = ['module', 'provider', None]

        # Run
        AnonymizedFaker.check_provider_function('BaseProvider', 'function_name')

        # Assert
        assert mock_getattr.call_args_list[0] == call(mock_faker.providers, 'BaseProvider')
        assert mock_getattr.call_args_list[1] == call('module', 'function_name')

    @patch('rdt.transformers.pii.anonymizer.faker')
    @patch('rdt.transformers.pii.anonymizer.getattr')
    def test_check_provider_function_other_providers(self, mock_getattr, mock_faker):
        """Test that ``getattr`` is being called with ``provider_name`` and ``function_name``.

        Mock:
            - Mock the ``getattr`` from Python to ensure that is being called with the input.
            - Mock faker and ensure that ``getattr`` is being called with ``faker.providers``.
        """
        # Setup
        mock_getattr.side_effect = ['module', 'provider_class', None]

        # Run
        AnonymizedFaker.check_provider_function('provider_name', 'function_name')

        # Assert
        assert mock_getattr.call_args_list[0] == call(mock_faker.providers, 'provider_name')
        assert mock_getattr.call_args_list[1] == call('module', 'Provider')
        assert mock_getattr.call_args_list[2] == call('provider_class', 'function_name')

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
        with pytest.raises(Error, match=expected_message):
            AnonymizedFaker.check_provider_function('TestProvider', 'TestFunction')

    def test__function(self):
        """Test that `_function`.

        The method `_function` should return a call from the `instance.faker.provider.<function>`.

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
        function.return_value = 1
        instance.faker.number = function
        instance.function_name = 'number'
        instance.function_kwargs = {'type': 'int'}

        # Run
        result = AnonymizedFaker._function(instance)

        # Assert
        function.assert_called_once_with(type='int')
        assert result == 1

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
            - the ``instance.missing_value_replacement`` is ``None``.
            - the ``instance.model_missing_values`` is ``False``
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
        assert instance.missing_value_replacement is None
        assert not instance.model_missing_values
        assert instance.locales is None
        assert mock_faker.Faker.called_once_with(None)

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
            - the ``instance.missing_value_replacement`` is ``None``.
            - the ``instance.model_missing_values`` is ``False``
            - ``check_provider_function`` has been called once with ``CreditCard`` and
              ``credit_card_full``.
            - the ``instance._function`` is ``instance.faker.credit_card_full``.
        """
        # Run
        instance = AnonymizedFaker(
            provider_name='credit_card',
            function_name='credit_card_full',
            function_kwargs={
                'type': 'visa'
            },
            locales=['en_US', 'fr_FR']
        )

        # Assert
        mock_check_provider_function.assert_called_once_with('credit_card', 'credit_card_full')
        assert instance.provider_name == 'credit_card'
        assert instance.function_name == 'credit_card_full'
        assert instance.function_kwargs == {'type': 'visa'}
        assert instance.missing_value_replacement is None
        assert not instance.model_missing_values
        assert instance.locales == ['en_US', 'fr_FR']
        assert mock_faker.Faker.called_once_with(['en_US', 'fr_FR'])

    def test___init__no_function_name(self):
        """Test the instantiation of the transformer with custom parameters.

        Test that the transformer raises an error when no function name is provided.

        Raises:
            - Error.
        """
        # Run / Assert
        expected_message = (
            'Please specify the function name to use from the '
            "'credit_card' provider."
        )
        with pytest.raises(Error, match=expected_message):
            AnonymizedFaker(provider_name='credit_card', locales=['en_US', 'fr_FR'])

    def test_get_output_sdtypes(self):
        """Test the ``get_output_sdtypes``.

        Setup:
            - initialize a ``AnonymizedFaker`` transformer with default values.

        Output:
            - the ``output_sdtypes`` returns an empty dictionary.
        """
        # Setup
        transformer = AnonymizedFaker()
        transformer.column_prefix = 'a#b'

        # Run
        output = transformer.get_output_sdtypes()

        # Assert
        expected = {}
        assert output == expected

    def test_get_output_sdtypes_model_missing_values(self):
        """Test the ``get_output_sdtypes`` method when a null column is created.

        Setup:
            - initialize a ``AnonymizedFaker`` transformer which:
                - sets ``self.null_transformer`` to a ``NullTransformer`` where
                ``self.model_missing_values`` is True.
                - sets ``self.column_prefix`` to a string.

        Output:
            - An ``output_sdtypes`` dictionary is being returned with the ``self.column_prefix``
              added to the beginning of the keys.
        """
        # Setup
        transformer = AnonymizedFaker()
        transformer.null_transformer = NullTransformer(missing_value_replacement='fill')
        transformer.null_transformer._model_missing_values = True
        transformer.column_prefix = 'a#b'

        # Run
        output = transformer.get_output_sdtypes()

        # Assert
        expected = {
            'a#b.is_null': 'float'
        }
        assert output == expected

    @patch('rdt.transformers.pii.anonymizer.NullTransformer')
    def test__fit(self, mock_null_transformer):
        """Test the ``_fit`` method.

        Validate that the ``_fit`` method uses the ``NullTransformer`` to parse the data
        and learn the length of it.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.
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
        transformer = AnonymizedFaker()

        columns_data = pd.Series(['1', '2', '3'])

        # Run
        transformer._fit(columns_data)

        # Assert
        mock_null_transformer.assert_called_once_with(None, False)
        mock_null_transformer.return_value.fit.called_once_with(columns_data)
        assert transformer.data_length == 3

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method returns ``None`` when the ``NullTransformer``
        does not model the missing values.

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

    def test__transform_model_missing_values(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method uses the ``NullTransformer`` instance to
        transform the data.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.
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
        instance = AnonymizedFaker()
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

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method calls the ``instance._function`` with
        the ``instance.function_kwargs`` the ``instance.data_length`` amount of times.

        Setup:
            - Initialize a ``AnonymizedFaker`` transformer.
            - Mock the ``null_transformer`` of the instance.
            - Mock the return value of the ``null_transformer.reverse_transform``.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - the output of ``null_transformer.reverse_transform``.
        """
        # Setup
        instance = AnonymizedFaker()
        instance.null_transformer = Mock()
        instance.null_transformer.models_missing_values.return_value = False
        instance.data_length = 3
        function = Mock()
        function.side_effect = ['a', 'b', 'c']

        instance._function = function
        instance.null_transformer.reverse_transform.return_value = np.array(['a', 'b', 'c'])

        # Run
        result = instance._reverse_transform(None)

        # Assert
        expected_null_call = np.array(['a', 'b', 'c'])
        null_call = instance.null_transformer.reverse_transform.call_args_list[0][0][0]
        np.testing.assert_array_equal(null_call, expected_null_call)
        assert function.call_args_list == [call(), call(), call()]
        np.testing.assert_array_equal(result, np.array(['a', 'b', 'c']))

    def test__reverse_transform_models_missing_values(self):
        """Test the ``_reverse_transform`` method.

        Validate that the ``_reverse_transform`` method uses the ``instance._function``
        to generate values within the range of the ``instance.data_length``.

        Setup:
            - Mock the instance of ``AnonymizedFaker``.
            - Mock the ``instance.null_transformer.reverse_transform`` return value.

        Input:
            - ``pd.DataFrame`` with a column ``is_null`` that contains numeric values.

        Output:
            - The mocked return value of the ``null_transformer.reverse_transform``.

        Side Effects:
            - The ``instance._function`` has been called ``instance.data_length`` with
              the ``function_args`` as keyword args.
        """
        # Setup
        columns_data = pd.DataFrame({
            'is_null': [0, 1, 0]
        })
        instance = Mock()
        instance.null_transformer.reverse_transform.return_value = pd.Series([
            'a',
            np.nan,
            'c',
        ])
        instance.get_output_columns.return_value = ['is_null']
        function = Mock()
        function.side_effect = [1, 2, 3]
        instance._function = function
        instance.data_length = 3
        instance.function_kwargs = {
            'type': 'a'
        }

        # Run
        output = AnonymizedFaker._reverse_transform(instance, columns_data)

        # Assert
        expected_output = pd.Series([
            'a',
            np.nan,
            'c'
        ])
        pd.testing.assert_series_equal(expected_output, output)

        expected_call = np.array([
            [1, 0],
            [2, 1],
            [3, 0]
        ])
        called_arg = instance.null_transformer.reverse_transform.call_args[0][0]
        np.testing.assert_array_equal(expected_call, called_arg)

        expected_function_calls = [call(), call(), call()]
        assert function.call_args_list == expected_function_calls

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
        instance = AnonymizedFaker('credit_card', 'credit_card_full', model_missing_values=True)

        # Run
        res = repr(instance)

        # Assert
        expected_res = (
            "AnonymizedFaker(provider_name='credit_card', function_name='credit_card_full', "
            'model_missing_values=True)'
        )

        assert res == expected_res
