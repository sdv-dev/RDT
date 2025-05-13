"""Personal Identifiable Information Anonymizer."""

import hashlib
import importlib
import inspect
import warnings
from collections.abc import Iterable
from copy import deepcopy
from operator import attrgetter

import faker
import numpy as np
import pandas as pd

from rdt.errors import TransformerInputError, TransformerProcessingError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.categorical import LabelEncoder
from rdt.transformers.utils import (
    _get_cardinality_frequency,
    _handle_enforce_uniqueness_and_cardinality_rule,
    _sample_repetitions,
)


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
        cardinality_rule (str):
            If ``'unique'`` enforce that every created value is unique.
            If ``'match'`` match the cardinality of the data seen during fit.
            If set to 'scale', the generated data will match the number of repetitions that
              each value is allowed to have.
            If ``None`` do not consider cardinality.
            Defaults to ``None``.
        enforce_uniqueness (bool):
            **DEPRECATED** Whether or not to ensure that the new anonymized data is all unique.
            If it isn't possible to create the requested number of rows, then an error will be
            raised.
            Defaults to ``False``.
        missing_value_generation (str or None):
            The way missing values are being handled. There are two strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``None``: Don't learn anything during fit. Then during reverse transform,
                  don't create any missing values.

    """

    # pylint: disable=too-many-instance-attributes

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
            provider_name = self.provider_name
            if self.provider_name.endswith(f'.{locale}'):
                provider_name = self.provider_name.replace(f'.{locale}', '')

            spec = importlib.util.find_spec(f'faker.providers.{provider_name}.{locale}')
            if spec is None and locale != 'en_US':
                missed_locales.append(locale)

        if missed_locales:
            warnings.warn(
                f"Locales {missed_locales} do not support provider '{self.provider_name}' "
                f"and function '{self.function_name}'.\nIn place of these locales, 'en_US' will "
                'be used instead. Please refer to the localized provider docs for more '
                'information: https://faker.readthedocs.io/en/master/locales.html'
            )

    def __init__(
        self,
        provider_name=None,
        function_name=None,
        function_kwargs=None,
        locales=None,
        cardinality_rule=None,
        enforce_uniqueness=None,
        missing_value_generation='random',
    ):
        super().__init__()
        self._data_cardinality_scale = None
        self._remaining_samples = {'value': None, 'repetitions': 0}
        self._data_cardinality = None
        self.data_length = None
        self.enforce_uniqueness = enforce_uniqueness
        self.cardinality_rule = cardinality_rule.lower() if cardinality_rule else None
        self.cardinality_rule = _handle_enforce_uniqueness_and_cardinality_rule(
            enforce_uniqueness, cardinality_rule
        )

        self.provider_name = provider_name if provider_name else 'BaseProvider'
        if self.provider_name != 'BaseProvider' and function_name is None:
            raise TransformerInputError(
                f"Please specify the function name to use from the '{self.provider_name}' provider."
            )

        self.function_name = function_name if function_name else 'lexify'
        self.function_kwargs = deepcopy(function_kwargs) if function_kwargs else {}
        self.check_provider_function(self.provider_name, self.function_name)
        self.output_properties = {None: {'next_transformer': None}}

        self._faker_random_seed = None
        self.locales = locales
        self.faker = faker.Faker(self.locales)
        if self.provider_name != 'BaseProvider' and self.locales:
            self._check_locales()

        if missing_value_generation not in ['random', None]:
            raise TransformerInputError(
                f"Missing value generation '{missing_value_generation}' is not supported "
                "for AnonymizedFaker. Please use either 'random' or None."
            )

        self.missing_value_generation = missing_value_generation
        self._nan_frequency = 0.0
        self._unique_categories = None

    @classmethod
    def get_supported_sdtypes(cls):
        """Return the supported sdtypes by the transformer.

        Returns:
            list:
                Accepted input sdtypes of the transformer.
        """
        unsupported_sdtypes = {
            'numerical',
            'datetime',
            'categorical',
            'boolean',
            None,
        }
        all_sdtypes = {cls.INPUT_SDTYPE}
        for transformer in BaseTransformer.get_subclasses():
            if not issubclass(transformer, cls):
                all_sdtypes.update(transformer.get_supported_sdtypes())

        supported_sdtypes = all_sdtypes - unsupported_sdtypes
        return list(supported_sdtypes)

    def reset_randomization(self):
        """Create a new ``Faker`` instance."""
        super().reset_randomization()
        self.faker = faker.Faker(self.locales)
        self.faker.seed_instance(self._faker_random_seed)

        if hasattr(self, 'cardinality_rule') and self.cardinality_rule == 'scale':
            self._remaining_samples['repetitions'] = 0
            np.random.seed(self.random_seed)

    def _function(self):
        """Return the result of calling the ``faker`` function."""
        try:
            if self.cardinality_rule in {'unique', 'match', 'scale'}:
                faker_attr = self.faker.unique
            else:
                faker_attr = self.faker

        except AttributeError:
            faker_attr = self.faker.unique if self.enforce_uniqueness else self.faker

        result = getattr(faker_attr, self.function_name)(**self.function_kwargs)
        if isinstance(result, Iterable) and not isinstance(result, str):
            result = ', '.join(map(str, result))

        return result

    def _fallback_function(self):
        """Return the result of calling a fallback ``faker`` function."""
        return self.faker.unique.bothify(text='??????')

    def _set_faker_seed(self, data):
        hash_value = self.get_input_column()
        for value in data.head(5):
            hash_value += str(value)

        hash_value = int(hashlib.sha256(hash_value.encode('utf-8')).hexdigest(), 16)
        self._faker_random_seed = hash_value % ((2**32) - 1)  # maximum value for a seed
        self.faker.seed_instance(self._faker_random_seed)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self._set_faker_seed(data)
        self.data_length = len(data)
        if self.missing_value_generation == 'random':
            self._nan_frequency = data.isna().sum() / len(data) if len(data) > 0 else 0.0

        if self.cardinality_rule == 'match':
            # remove nans from data
            self._data_cardinality = len(data.dropna().unique())

        if self.cardinality_rule == 'scale':
            sorted_counts, sorted_frequencies = _get_cardinality_frequency(data.dropna())
            self._data_cardinality_scale = {
                'num_repetitions': sorted_counts,
                'frequency': sorted_frequencies,
            }

    def _transform(self, _data):
        """Drop the input column by returning ``None``."""
        return None

    def _generate_cardinality_scale_values(self, remaining_samples):
        """Generate sampled values while ensuring each unique category appears at least once."""
        if self._remaining_samples['repetitions'] >= remaining_samples:
            self._remaining_samples['repetitions'] -= remaining_samples
            return [self._remaining_samples['value']] * remaining_samples

        samples = [self._remaining_samples['value']] * self._remaining_samples['repetitions']
        self._remaining_samples['repetitions'] = 0

        while len(samples) < remaining_samples:
            new_samples, self._remaining_samples = _sample_repetitions(
                remaining_samples - len(samples),
                self._function(),
                self._data_cardinality_scale.copy(),
                self._remaining_samples.copy(),
            )
            samples.extend(new_samples)

        return np.array(samples, dtype=object)

    def _get_unique_categories(self, samples):
        return np.array([self._function() for _ in range(samples)], dtype=object)

    def _generate_cardinality_match_values(self, remaining_samples):
        """Generate sampled values while ensuring each unique category appears at least once."""
        # Backwards compatibility requires us to generate the values at this point
        if self._unique_categories is None:
            self._unique_categories = self._get_unique_categories(self._data_cardinality)

        unique_categories = np.array(self._unique_categories)
        if remaining_samples <= len(unique_categories):
            return np.random.choice(unique_categories, remaining_samples, replace=False)

        # Ensure all unique categories appear at least once
        extra_samples_needed = remaining_samples - len(unique_categories)
        extra_samples = np.random.choice(unique_categories, extra_samples_needed, replace=True)

        return np.concatenate((unique_categories, extra_samples))

    def _reverse_transform_with_fallback(self, sample_size):
        try:
            reverse_transformed = []
            for _ in range(sample_size):
                reverse_transformed.append(self._function())

        except faker.exceptions.UniquenessException:
            warnings.warn(
                f"Unable to generate enough unique values for column '{self.get_input_column()}' "
                'in a human-readable format. Additional values may be created randomly.'
            )
            remaining_samples = sample_size - len(reverse_transformed)
            for _ in range(remaining_samples):
                reverse_transformed.append(self._fallback_function())

        return np.array(reverse_transformed, dtype=object)

    def _calculate_num_nans(self, sample_size):
        """Calculate the number of NaN values to generate."""
        if self.missing_value_generation == 'random':
            return int(self._nan_frequency * sample_size)

        return 0

    def _generate_nans(self, num_nans):
        """Generate an array of NaN values."""
        return np.full(num_nans, np.nan, dtype=object)

    def _reverse_transform_cardinality_rules(self, sample_size):
        """Reverse transform the data when the cardinality rule is 'match' or 'scale'."""
        num_nans = self._calculate_num_nans(sample_size)
        reverse_transformed = self._generate_nans(num_nans)

        if sample_size <= num_nans:
            return reverse_transformed

        remaining_samples = sample_size - num_nans
        if self.cardinality_rule == 'match':
            sampled_values = self._generate_cardinality_match_values(remaining_samples)
        else:
            sampled_values = self._generate_cardinality_scale_values(remaining_samples)

        reverse_transformed = np.concatenate([reverse_transformed, sampled_values])
        np.random.shuffle(reverse_transformed)

        return reverse_transformed

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

        if hasattr(self, 'cardinality_rule') and self.cardinality_rule in {'match', 'scale'}:
            reverse_transformed = self._reverse_transform_cardinality_rules(sample_size)
        else:
            reverse_transformed = self._reverse_transform_with_fallback(sample_size)

        if self.missing_value_generation == 'random' and pd.notna(reverse_transformed).all():
            num_nans = int(self._nan_frequency * sample_size)
            nan_indices = np.random.choice(sample_size, num_nans, replace=False)
            reverse_transformed[nan_indices] = np.nan

        return reverse_transformed

    def _set_fitted_parameters(
        self,
        column_name,
        nan_frequency=0.0,
        cardinality=None,
        cardinality_scale=None,
    ):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column to use for the transformer.
            nan_frequency (float):
                The fraction of values that should be replaced with nan values
                if self.missing_value_generation is 'random'.
            cardinality (int or None):
                The number of unique values to generate if cardinality rule is set to
                'match'.
            cardinality_scale (dict or None):
                The frequency of each number of repetitions in the data:
                {
                    'num_repetitions': list of int,
                    'frequency': list of float,
                }
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        if self.cardinality_rule == 'match':
            if not cardinality:
                raise TransformerInputError(
                    'Cardinality "match" rule must specify a cardinality value.'
                )

        if self.cardinality_rule == 'scale':
            if not cardinality_scale:
                raise TransformerInputError(
                    'Cardinality "scale" rule must specify a cardinality value.'
                )

        self._data_cardinality = cardinality
        self._data_cardinality_scale = cardinality_scale
        self._nan_frequency = nan_frequency

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
        keys.remove('enforce_uniqueness')
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
        warnings.warn(
            (
                'You are saving the mapping information, which includes the original data. '
                'Sharing this object with others will also give them access to the original data '
                'used with this transformer.'
            )
        )

        return self.__dict__

    def __init__(
        self,
        provider_name=None,
        function_name=None,
        function_kwargs=None,
        locales=None,
    ):
        super().__init__(
            provider_name=provider_name,
            function_name=function_name,
            function_kwargs=function_kwargs,
            locales=locales,
            cardinality_rule='unique',
        )
        self._mapping_dict = {}
        self._reverse_mapping_dict = {}
        self.output_properties = {
            None: {
                'sdtype': 'categorical',
                'next_transformer': LabelEncoder(add_noise=True),
            }
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

        generated_values = self._reverse_transform_with_fallback(unique_data_length)
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
