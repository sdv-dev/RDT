"""Transformers for ID data."""

import logging
import warnings

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.utils import (
    _handle_enforce_uniqueness_and_cardinality_rule,
    strings_from_regex,
)

LOGGER = logging.getLogger(__name__)


class IDGenerator(BaseTransformer):
    """Generate an ID column.

    This transformer generates an ID column based on a given prefix, starting value and suffix.

    Args:
            prefix (str):
                Prefix of the generated IDs column.
                Defaults to ``None``.
            starting_value (int):
                Starting value of the generated IDs column.
                Defaults to ``0``.
            suffix (str):
                Suffix of the generated IDs column.
                Defaults to ``None``.
    """

    IS_GENERATOR = True
    INPUT_SDTYPE = 'id'
    SUPPORTED_SDTYPES = ['id', 'text']

    def __init__(self, prefix=None, starting_value=0, suffix=None):
        super().__init__()
        self.prefix = prefix
        self.starting_value = starting_value
        self.suffix = suffix
        self._counter = 0
        self.output_properties = {None: {'next_transformer': None}}

    def reset_randomization(self):
        """Reset the sampling _counter."""
        self._counter = 0

    def _fit(self, data):
        pass

    def _transform(self, _data):
        """Drop the input column by returning ``None``."""
        return None

    def _reverse_transform(self, data):
        """Generate new id column.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pd.Series
        """
        start = self.starting_value + self._counter
        prefix_str = self.prefix if self.prefix is not None else ''
        suffix_str = self.suffix if self.suffix is not None else ''

        values = [f'{prefix_str}{start + idx}{suffix_str}' for idx in range(len(data))]
        self._counter += len(data)

        return pd.Series(values)


class RegexGenerator(BaseTransformer):
    """RegexGenerator transformer.

    This transformer will drop a column and regenerate it with the previously specified
    ``regex`` format.

    Args:
        regex_format (str):
            String representing the regex function.
        enforce_uniqueness (bool):
            **DEPRECATED** Whether or not to ensure that the new generated data is all unique.
            If it isn't possible to create the requested number of rows, then an error will
            be raised. Defaults to ``None``.
        cardinality_rule (str):
            Rule that the generated data must follow.
            - If set to 'unique', the generated data must be unique.
            - If set to 'match', the generated data must have the exact same cardinality
              (# of unique values) as the real data.
            - If set to ``None``, then the generated data may contain duplicates.
            Defaults to ``None``.
        generation_order (str):
            String defining how to generate the output. If set to ``alphanumeric``, it will
            generate the output in alphanumeric order (ie. 'aaa', 'aab' or '1', '2'...). If
            set to ``scrambled``, the the output will be scrambled in order. Defaults to
            ``alphanumeric``.
    """

    IS_GENERATOR = True
    INPUT_SDTYPE = 'id'
    SUPPORTED_SDTYPES = ['id', 'text']

    def __getstate__(self):
        """Remove the generator when pickling."""
        state = self.__dict__.copy()
        state.pop('generator')
        return state

    def __setstate__(self, state):
        """Set the generator when pickling."""
        generator_size = state.get('generator_size')
        generated = state.get('generated')
        generator, size = strings_from_regex(state.get('regex_format'))
        if generator_size is None:
            state['generator_size'] = size
        if generated is None:
            state['generated'] = 0

        if generated:
            for _ in range(generated):
                next(generator)

        state['generator'] = generator
        self.__dict__ = state

    def __init__(
        self,
        regex_format='[A-Za-z]{5}',
        cardinality_rule=None,
        generation_order='alphanumeric',
        enforce_uniqueness=None,
    ):
        super().__init__()
        self.output_properties = {None: {'next_transformer': None}}
        self.regex_format = regex_format
        self.cardinality_rule = _handle_enforce_uniqueness_and_cardinality_rule(
            enforce_uniqueness, cardinality_rule
        )
        self._data_cardinality = None
        self._unique_regex_values = None
        self.data_length = None
        self.generator = None
        self.generator_size = None
        self.generated = None
        if generation_order not in ['alphanumeric', 'scrambled']:
            raise ValueError("generation_order must be one of 'alphanumeric' or 'scrambled'.")

        self.generation_order = generation_order

    def reset_randomization(self):
        """Create a new generator and reset the generated values counter."""
        super().reset_randomization()
        self.generator, self.generator_size = strings_from_regex(self.regex_format)
        self.generated = 0

    def _generate_fallback_samples(self, num_samples, template_samples):
        """Generate values such that they are all unique, disregarding the regex."""
        try:
            # Integer-based fallback: attempt to convert the last template sample to an integer
            # and then generate values in a sequential manner.
            start = int(template_samples[-1]) + 1
            return [str(i) for i in range(start, start + num_samples)]

        except ValueError:
            # String-based fallback: if the integer conversion fails, it uses the template
            # samples as a base and appends a counter to make each value unique.
            counter = 0
            samples = []
            while num_samples > 0:
                samples.extend([f'{i}({counter})' for i in template_samples[:num_samples]])
                num_samples -= len(template_samples)
                counter += 1

            return samples

    def _generate_unique_regex_values(self):
        regex_values = []
        try:
            while len(regex_values) < self._data_cardinality:
                regex_values.append(next(self.generator))
        except (RuntimeError, StopIteration):
            fallback_samples = self._generate_fallback_samples(
                num_samples=self._data_cardinality - len(regex_values),
                template_samples=regex_values,
            )
            regex_values.extend(fallback_samples)

        return regex_values

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.reset_randomization()
        self.data_length = len(data)

        if self.cardinality_rule == 'match':
            is_nan = int(pd.isna(data).any())  # nans count as a unique value
            self._data_cardinality = data.nunique() + is_nan
            self._unique_regex_values = self._generate_unique_regex_values()

    def _transform(self, _data):
        """Drop the input column by returning ``None``."""
        return None

    def _warn_not_enough_unique_values(self, sample_size, unique_condition, match_cardinality):
        """Warn the user that the regex cannot generate enough unique values.

        Args:
            sample_size (int):
                Number of samples to be generated.
            unique_condition (bool):
                Whether or not to enforce uniqueness.
            match_cardinality (bool):
                Whether or not to match the cardinality of the data.
        """
        warned = False
        warn_msg = (
            f"The regex for '{self.get_input_column()}' can only generate "
            f'{int(self.generator_size)} unique values. Additional values may not exactly '
            'follow the provided regex.'
        )
        if sample_size > self.generator_size:
            if unique_condition:
                warnings.warn(warn_msg)
                warned = True
            else:
                LOGGER.info(
                    "The data has %s rows but the regex for '%s' can only create %s unique values."
                    " Some values in '%s' may be repeated.",
                    sample_size,
                    self.get_input_column(),
                    self.generator_size,
                    self.get_input_column(),
                )

        remaining = self.generator_size - self.generated
        if sample_size > remaining and unique_condition and not warned:
            warnings.warn(
                f'The regex generator is not able to generate {sample_size} new unique '
                f'values (only {max(remaining, 0)} unique values left).'
            )

        if match_cardinality:
            if self._data_cardinality > sample_size:
                warnings.warn(
                    f'Only {sample_size} values can be generated. Cannot match the cardinality '
                    f'of the data, it requires {self._data_cardinality} values.'
                )
            if sample_size > self.generator_size and self._data_cardinality > self.generator_size:
                warnings.warn(warn_msg)

    def _generate_as_many_as_possible(self, num_samples):
        """Generate samples.

        Generate values following the regex until either the sample size is reached or
        the generator is exhausted.
        """
        generated_values = []
        try:
            while len(generated_values) < num_samples:
                generated_values.append(next(self.generator))
                self.generated += 1
        except (RuntimeError, StopIteration):
            pass

        return generated_values

    def _generate_num_samples(self, num_samples, template_samples):
        """Generate num_samples values from template_samples.

        Eg: num_samples = 5, template_samples = ['a', 'b']
        The output will be ['a', 'b', 'a', 'b', 'a']
        """
        if num_samples <= 0:
            return []

        repeats = num_samples // len(template_samples) + 1
        return np.tile(template_samples, repeats)[:num_samples].tolist()

    def _generate_match_cardinality(self, num_samples):
        """Generate values until the sample size is reached, while respecting the cardinality."""
        template_samples = self._unique_regex_values[:num_samples]
        samples = self._generate_num_samples(num_samples - len(template_samples), template_samples)

        return template_samples + samples

    def _generate_samples(self, num_samples, match_cardinality_values, unique_condition):
        """Generate samples until the sample size is reached."""
        if match_cardinality_values is not None:
            return self._generate_match_cardinality(num_samples)

        # If there aren't enough values left in the generator, reset it
        if num_samples > self.generator_size - self.generated:
            self.reset_randomization()

        samples = self._generate_as_many_as_possible(num_samples)
        num_samples -= len(samples)
        if num_samples > 0:
            if unique_condition:
                new_samples = self._generate_fallback_samples(num_samples, samples)
            else:
                new_samples = self._generate_num_samples(num_samples, samples)
            samples.extend(new_samples)

        return samples

    def _reverse_transform(self, data):
        """Generate new data using the provided ``regex_format``.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if hasattr(self, 'cardinality_rule'):
            unique_condition = self.cardinality_rule == 'unique'
        else:
            unique_condition = self.enforce_uniqueness

        if hasattr(self, '_unique_regex_values'):
            match_cardinality_values = self._unique_regex_values
        else:
            match_cardinality_values = None

        if data is not None and len(data):
            num_samples = len(data)
        else:
            num_samples = self.data_length

        match_condition = match_cardinality_values is not None
        self._warn_not_enough_unique_values(num_samples, unique_condition, match_condition)
        samples = self._generate_samples(num_samples, match_cardinality_values, unique_condition)

        if getattr(self, 'generation_order', 'alphanumeric') == 'scrambled':
            np.random.shuffle(samples)

        return np.array(samples, dtype=object)
