"""Test Personal Identifiable Information Transformer using Faker."""

from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.null import NullTransformer
from rdt.transformers.pii.faker import PIIFaker


class TestPIIFaker:
    """Test class for ``PIIFaker``."""

    @patch('rdt.transformers.pii.faker.faker')
    @patch('rdt.transformers.pii.faker.getattr')
    def test_check_provider_function(self, mock_getattr, mock_faker):
        """Test that ``getattr`` is being called with ``provider_name`` and ``function_name``.

        Mock:
            - Mock the ``getattr`` from Python to ensure that is being called with the input.
            - Mock faker and ensure that ``getattr`` is being called with ``faker.providers``.
        """
        # Setup
        mock_getattr.side_effect = ['provider_class', None]

        # Run
        PIIFaker.check_provider_function('provider_name', 'function_name')

        # Assert
        assert mock_getattr.call_args_list[0] == call(mock_faker.providers, 'provider_name')
        assert mock_getattr.call_args_list[1] == call('provider_class', 'function_name')

    def test_check_provider_function_raise_attribute_error(self):
        """Test that ``check_provider_function`` raises an ``AttributeError``.

        Test when the provider class is not in the ``faker.providers`` an ``AttributeError`` is
        being raised or when the ``function`` is not within the ``provider``.
        """
        # Setup
        expected_provider_message = "module 'faker.providers' has no attribute 'TestProvider'"
        expected_function_message = "type object 'BaseProvider' has no attribute 'TestFunction'"

        # Run
        with pytest.raises(AttributeError, match=expected_provider_message):
            PIIFaker.check_provider_function('TestProvider', 'TestFunction')

        with pytest.raises(AttributeError, match=expected_function_message):
            PIIFaker.check_provider_function('BaseProvider', 'TestFunction')

    @patch('rdt.transformers.pii.faker.faker')
    @patch('rdt.transformers.pii.faker.PIIFaker.check_provider_function')
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
        instance = PIIFaker()

        # Assert
        mock_check_provider_function.assert_called_once_with('BaseProvider', 'lexify')
        assert instance.provider_name == 'BaseProvider'
        assert instance.function_name == 'lexify'
        assert instance.function_kwargs == {}
        assert instance.missing_value_replacement is None
        assert not instance.model_missing_values
        assert instance.locales is None
        assert mock_faker.Faker.called_once_with(None)
        assert instance._function == mock_faker.Faker.return_value.lexify

    @patch('rdt.transformers.pii.faker.faker')
    @patch('rdt.transformers.pii.faker.PIIFaker.check_provider_function')
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
        instance = PIIFaker(
            provider_name='CreditCard',
            function_name='credit_card_full',
            function_kwargs={
                'type': 'visa'
            },
            locales=['en_US', 'fr_FR']
        )

        # Assert
        mock_check_provider_function.assert_called_once_with('CreditCard', 'credit_card_full')
        assert instance.provider_name == 'CreditCard'
        assert instance.function_name == 'credit_card_full'
        assert instance.function_kwargs == {'type': 'visa'}
        assert instance.missing_value_replacement is None
        assert not instance.model_missing_values
        assert instance.locales == ['en_US', 'fr_FR']
        assert mock_faker.Faker.called_once_with(['en_US', 'fr_FR'])
        assert instance._function == mock_faker.Faker.return_value.credit_card_full

    def test_get_output_types(self):
        """Test the ``get_output_types``.

        Setup:
            - initialize a ``PIIFaker`` transformer which:

        Output:
            - the ``output_types`` returns an empty dictionary.
        """
        # Setup
        transformer = PIIFaker()
        transformer.column_prefix = 'a#b'

        # Run
        output = transformer.get_output_types()

        # Assert
        expected = {}
        assert output == expected

    def test_get_output_types_model_missing_values(self):
        """Test the ``get_output_types`` method when a null column is created.

        Setup:
            - initialize a ``PIIFaker`` transformer which:
                - sets ``self.null_transformer`` to a ``NullTransformer`` where
                ``self.model_missing_values`` is True.
                - sets ``self.column_prefix`` to a string.

        Output:
            - An ``output_types`` dictionary is being returned with the ``self.column_prefix``
              added to the beginning of the keys.
        """
        # Setup
        transformer = PIIFaker()
        transformer.null_transformer = NullTransformer(missing_value_replacement='fill')
        transformer.null_transformer._model_missing_values = True
        transformer.column_prefix = 'a#b'

        # Run
        output = transformer.get_output_types()

        # Assert
        expected = {
            'a#b.is_null': 'float'
        }
        assert output == expected

    @patch('rdt.transformers.pii.faker.NullTransformer')
    def test__fit(self, mock_null_transformer):
        """Test the ``_fit`` method.

        Validate that the ``_fit`` method uses the ``NullTransformer`` to parse the data
        and learn the length of it.

        Setup:
            - Initialize a ``PIIFaker`` transformer.
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
        transformer = PIIFaker()

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
            - Initialize a ``PIIFaker`` transformer.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - ``None``.
        """
        # Setup
        columns_data = pd.Series([1, 2, 3])
        instance = PIIFaker()

        # Run
        result = instance._transform(columns_data)

        # Assert
        assert result is None

    def test__transform_model_missing_values(self):
        """Test the ``_transform`` method.

        Validate that the ``_transform`` method uses the ``NullTransformer`` instance to
        transform the data.

        Setup:
            - Initialize a ``PIIFaker`` transformer.
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
        instance = PIIFaker()
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
            - Initialize a ``PIIFaker`` transformer.
            - Mock the ``null_transformer`` of the instance.
            - Mock the return value of the ``null_transformer.reverse_transform``.

        Input:
            - ``pd.Series`` with three values.

        Output:
            - the output of ``null_transformer.reverse_transform``.
        """
        # Setup
        columns_data = pd.Series([1, 2, 3])
        instance = PIIFaker()
        instance.null_transformer = Mock()
        instance.null_transformer.models_missing_values.return_value = False
        instance.data_length = 3
        function = Mock()
        function.side_effect = ['a', 'b', 'c']

        instance._function = function
        instance.null_transformer.reverse_transform.return_value = np.array([
            'a',
            'b',
            'c',
        ])

        # Run
        result = instance._reverse_transform(columns_data)

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
            - Mock the instance of ``PIIFaker``.
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
        output = PIIFaker._reverse_transform(instance, columns_data)

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

        expected_function_calls = [call(type='a'), call(type='a'), call(type='a')]
        assert function.call_args_list == expected_function_calls
