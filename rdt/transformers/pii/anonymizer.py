"""Personal Identifiable Information Anonymizer."""

import hashlib
import importlib
import inspect
import warnings
from copy import deepcopy
from operator import attrgetter

import faker
import numpy as np

from rdt.errors import TransformerInputError, TransformerProcessingError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.categorical import LabelEncoder


class AnonymizedFaker(BaseTransformer):
    """Personal Identifiable Information Anonymizer using Faker.

    This transformer will drop a column and regenerate it with the previously specified
    ``Faker`` provider and ``function``.

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
        enforce_uniqueness (bool):
            Whether or not to ensure that the new anonymized data is all unique. If it isn't
            possible to create the requested number of rows, then an error will be raised.
            Defaults to ``False``.
    """

    IS_GENERATOR = True
    INPUT_SDTYPE = 'pii'

    @staticmethod
    def check_provider_function(provider_name, function_name):
        """Check that the provider and the function exist.

        Attempt to get the provider from ``faker.providers`` and then get the ``function``
        from the provider object. If one of them fails, it will raise an ``AttributeError``.

        Raises:
            ``AttributeError`` if the provider or the function is not found.
        """
        try:
            module = attrgetter(provider_name)(faker.providers)
            if provider_name.lower() == 'baseprovider':
                getattr(module, function_name)

            else:
                provider = getattr(module, 'Provider')
                getattr(provider, function_name)

        except AttributeError as exception:
            raise TransformerProcessingError(
                f"The '{provider_name}' module does not contain a function named "
                f"'{function_name}'.\nRefer to the Faker docs to find the correct function: "
                'https://faker.readthedocs.io/en/master/providers.html'
            ) from exception

    def _check_locales(self):
        """Check if the locales exist for the provided provider."""
        locales = self.locales if isinstance(self.locales, list) else [self.locales]
        missed_locales = []
        for locale in locales:
            spec = importlib.util.find_spec(f'faker.providers.{self.provider_name}.{locale}')
            if spec is None:
                missed_locales.append(locale)

        if missed_locales:
            warnings.warn(
                f"Locales {missed_locales} do not support provider '{self.provider_name}' "
                f"and function '{self.function_name}'.\nIn place of these locales, 'en_US' will "
                'be used instead. Please refer to the localized provider docs for more '
                'information: https://faker.readthedocs.io/en/master/locales.html'
            )

    def __init__(self, provider_name=None, function_name=None, function_kwargs=None,
                 locales=None, enforce_uniqueness=False):
        super().__init__()
        self.data_length = None
        self.enforce_uniqueness = enforce_uniqueness
        self.provider_name = provider_name if provider_name else 'BaseProvider'
        if self.provider_name != 'BaseProvider' and function_name is None:
            raise TransformerInputError(
                'Please specify the function name to use from the '
                f"'{self.provider_name}' provider."
            )

        self.function_name = function_name if function_name else 'lexify'
        self.function_kwargs = deepcopy(function_kwargs) if function_kwargs else {}
        self.check_provider_function(self.provider_name, self.function_name)
        self.output_properties = {None: {'next_transformer': None}}

        self._faker_random_seed = None
        self.locales = locales
        self.faker = faker.Faker(self.locales)
        if self.locales:
            self._check_locales()

    def reset_randomization(self):
        """Create a new ``Faker`` instance."""
        super().reset_randomization()
        self.faker = faker.Faker(self.locales)
        self.faker.seed_instance(self._faker_random_seed)

    def _function(self):
        """Return a callable ``faker`` function."""
        if self.enforce_uniqueness:
            return getattr(self.faker.unique, self.function_name)(**self.function_kwargs)

        return getattr(self.faker, self.function_name)(**self.function_kwargs)

    def _set_faker_seed(self, data):
        hash_value = self.get_input_column()
        for value in data.head(5):
            hash_value += str(value)

        hash_value = int(hashlib.sha256(hash_value.encode('utf-8')).hexdigest(), 16)
        self._faker_random_seed = hash_value % ((2 ** 32) - 1)  # maximum value for a seed
        self.faker.seed_instance(self._faker_random_seed)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self._set_faker_seed(data)
        self.data_length = len(data)

    def _transform(self, _data):
        """Drop the input column by returning ``None``."""
        return None

    def _reverse_transform(self, data):
        """Generate new anonymized data using a ``faker.provider.function``.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            np.array
        """
        if data is not None and len(data):
            sample_size = len(data)
        else:
            sample_size = self.data_length

        try:
            reverse_transformed = np.array([
                self._function()
                for _ in range(sample_size)
            ], dtype=object)
        except faker.exceptions.UniquenessException as exception:
            raise TransformerProcessingError(
                f'The Faker function you specified is not able to generate {sample_size} unique '
                'values. Please use a different Faker function for column '
                f"('{self.get_input_column()}')."
            ) from exception

        return reverse_transformed

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.get_name()
        custom_args = []
        args = inspect.getfullargspec(self.__init__)
        keys = args.args[1:]
        defaults = dict(zip(keys, args.defaults))
        instanced = {key: getattr(self, key) for key in keys}

        defaults['function_name'] = None
        for arg, value in instanced.items():
            if value and defaults[arg] != value and value != 'BaseProvider':
                value = f"'{value}'" if isinstance(value, str) else value
                custom_args.append(f'{arg}={value}')

        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'


class PseudoAnonymizedFaker(AnonymizedFaker):
    """Pseudo-anonymization Transformer using Faker.

    This transformer anonymizes values that can be traced back to the original input by using
    a mapping. The transformer will generate a mapping with the previously specified
    ``Faker`` provider and ``function``.

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
    """

    def __getstate__(self):
        """Return a dictionary representation of the instance and warn the user when pickling."""
        warnings.warn((
            'You are saving the mapping information, which includes the original data. '
            'Sharing this object with others will also give them access to the original data '
            'used with this transformer.'
        ))

        return self.__dict__

    def __init__(self, provider_name=None, function_name=None, function_kwargs=None, locales=None):
        super().__init__(
            provider_name=provider_name,
            function_name=function_name,
            function_kwargs=function_kwargs,
            locales=locales,
            enforce_uniqueness=True
        )
        self._mapping_dict = {}
        self._reverse_mapping_dict = {}
        self.output_properties = {
            None: {'sdtype': 'categorical', 'next_transformer': LabelEncoder(add_noise=True)}
        }

    def get_mapping(self):
        """Return the mapping dictionary."""
        return deepcopy(self._mapping_dict)

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Generate a ``_mapping_dict`` and a ``_reverse_mapping_dict`` for each
        value in the provided ``columns_data`` using the ``Faker`` provider and
        ``function``.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self._set_faker_seed(columns_data)
        unique_values = columns_data[columns_data.notna()].unique()
        unique_data_length = len(unique_values)
        try:
            generated_values = [self._function() for _ in range(unique_data_length)]
        except faker.exceptions.UniquenessException as exception:
            raise TransformerProcessingError(
                'The Faker function you specified is not able to generate '
                f'{unique_data_length} unique values. Please use a different '
                'Faker function for this column.'
            ) from exception

        generated_values = list(set(generated_values))
        self._mapping_dict = dict(zip(unique_values, generated_values))
        self._reverse_mapping_dict = dict(zip(generated_values, unique_values))

    def _transform(self, columns_data):
        """Replace each category with a numerical representation.

        Map the input ``columns_data`` using the previously generated values for each one.
        If the  ``columns_data`` contain unknown values, an error will be raised with the
        unknown categories.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            pd.Series
        """
        unique_values = columns_data[columns_data.notna()].unique()
        new_values = list(set(unique_values) - set(self._mapping_dict))
        if new_values:
            new_values = [str(value) for value in new_values]
            if len(new_values) < 5:
                new_values = ', '.join(new_values)
                error_msg = (
                    'The data you are transforming has new, unexpected values '
                    f'({new_values}). Please fit the transformer again using this '
                    'new data.'
                )
            else:
                diff = len(new_values) - 5
                new_values = ', '.join(new_values[:5])
                error_msg = (
                    'The data you are transforming has new, unexpected values '
                    f'({new_values} and {diff} more). Please fit the transformer again '
                    'using this new data.'
                )

            raise TransformerProcessingError(error_msg)

        mapped_data = columns_data.map(self._mapping_dict)
        return mapped_data

    def _reverse_transform(self, columns_data):
        """Return the input data.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        return columns_data
