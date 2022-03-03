"""Personal Identifiable Information Anonymizer."""

from copy import deepcopy

import faker
import numpy as np

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class PIIAnonymizer(BaseTransformer):
    """Personal Identifiable Information Anonymizer using Faker.

    This transformer will drop a column and regenerate it with the previously specified
    ``Faker`` provider and ``function``. The transformer will also be able to handle nulls
    and regenerate null values if specified.

    Args:
        provider_name (str):
            The name of the provider in ``Faker``. If ``None`` the ``BaseProvider`` is used.
            Defaults to ``None``.
        function_name (str):
            The name of the function to use within the ``faker.provider``. Defaults to
            ``lexify``.
        function_kwargs (dict):
            Keyword args to pass into the ``function_name`` when being called.
        locales (list):
            List of localized providers to use instead of the global provider.
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are
            given, replace them with the corresponding aggregation. If ``None`` is given,
            do not replace them. Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """

    INPUT_TYPE = 'pii'
    null_transformer = None

    @staticmethod
    def check_provider_function(provider_name, function_name):
        """Check that the provider and the function exist.

        Attempt to get the provider from ``faker.providers`` and then get the ``function``
        from the provider object. If one of them fails, it will raise an ``AttributeError``.

        Raises:
            ``AttributeError`` if the provider or the function is not found.
        """
        module = getattr(faker.providers, provider_name)
        if provider_name == 'BaseProvider':
            getattr(module, function_name)

        else:
            provider = getattr(module, 'Provider')
            getattr(provider, function_name)

    def __init__(self, provider_name=None, function_name='lexify', function_kwargs=None,
                 locales=None, missing_value_replacement=None, model_missing_values=False):
        self.data_length = None
        self.provider_name = provider_name if provider_name else 'BaseProvider'
        self.function_name = function_name
        self.function_kwargs = deepcopy(function_kwargs) if function_kwargs else {}
        self.check_provider_function(self.provider_name, self.function_name)

        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values

        self.locales = locales
        self.faker = faker.Faker(locales)
        self._function = getattr(self.faker, function_name)

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        output_types = {}
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_types['is_null'] = 'float'

        return self._add_prefix(output_types)

    def _fit(self, columns_data):
        self.null_transformer = NullTransformer(
            self.missing_value_replacement,
            self.model_missing_values
        )
        self.null_transformer.fit(columns_data)
        self.data_length = len(columns_data)

    def _transform(self, columns_data):
        if self.null_transformer and self.null_transformer.models_missing_values():
            return self.null_transformer.transform(columns_data)[:, 1].astype(float)

        return None

    def _reverse_transform(self, columns_data):
        reverse_transformed = np.array([
            self._function(**self.function_kwargs)
            for _ in range(self.data_length)
        ], dtype=object)

        if self.null_transformer.models_missing_values():
            reverse_transformed = np.column_stack((reverse_transformed, columns_data))

        return self.null_transformer.reverse_transform(reverse_transformed)

    def __repr__(self):
        """Represent the class instance only with the modified default arguments."""
        printable = [
            'provider_name',
            'function_name',
            'function_kwargs',
            'locales',
            'missing_value_replacement',
            'model_missing_values',
        ]
        instance_args = {
            key: f"'{value}'" if isinstance(value, str) else value
            for key, value in self.__dict__.items()
            if key in printable and value
        }

        if self.provider_name == 'BaseProvider':
            instance_args.pop('provider_name')

        instance_args = ', '.join(f'{key}={value}' for key, value in instance_args.items())
        return f'PIIAnonymizer({instance_args})'
