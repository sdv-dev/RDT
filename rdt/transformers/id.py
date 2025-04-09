"""Transformers for ID data."""

import logging
import warnings

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.utils import (
    _get_cardinality_frequency,
    _handle_enforce_uniqueness_and_cardinality_rule,
    _sample_repetitions,
    fill_nan_with_none,
    strings_from_regex,
)

LOGGER = logging.getLogger(__name__)


class IndexGenerator(BaseTransformer):
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


class IDGenerator(IndexGenerator):
    """Deprecated class name for ``IndexGenerator``.

    Class to ensure backwards compatibility with previous versions of RDT.
    """

    def __init__(self, prefix=None, starting_value=0, suffix=None):
        warnings.warn(
            "The 'IDGenerator' has been renamed to 'IndexGenerator'. Please update the"
            'name to ensure compatibility with future versions of RDT.',
            FutureWarning,
        )
        super().__init__(prefix, starting_value, suffix)


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
            - If set to 'match', the generated data will have the exact same cardinality
              (number of unique values) as the real data.
            - If set to 'scale', the generated data will match the number of repetitions that
              each value is allowed to have.
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
        self.data_length = None
        self.generator = None
        if generation_order not in ['alphanumeric', 'scrambled']:
            raise ValueError("generation_order must be one of 'alphanumeric' or 'scrambled'.")

        self.generation_order = generation_order

        # Used when cardinality_rule is 'scale'
        self._data_cardinality_scale = None
        self._remaining_samples = {'value': None, 'repetitions': 0}

        # Used when cardinality_rule is 'match'
        self._data_cardinality = None
        self._unique_regex_values = None

        # Used otherwise
        self.generator_size = None
        self.generated = None

    def reset_randomization(self):
        """Create a new generator and reset the generated values counter."""
        super().reset_randomization()
        self.generator, self.generator_size = strings_from_regex(self.regex_format)
        self.generated = 0

        if hasattr(self, 'cardinality_rule') and self.cardinality_rule == 'scale':
            self._remaining_samples['repetitions'] = 0
            np.random.seed(self.random_seed)

    def _sample_fallback(self, num_samples, template_samples):
        """Sample num_samples values such that they are all unique, disregarding the regex."""
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
            while num_samples > len(samples):
                samples.extend([f'{i}({counter})' for i in template_samples[:num_samples]])
                counter += 1

            return samples[:num_samples]

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.reset_randomization()
        self.data_length = len(data)

        if hasattr(self, 'cardinality_rule'):
            data = fill_nan_with_none(data)
            if self.cardinality_rule == 'match':
                self._data_cardinality = data.nunique(dropna=False)

            elif self.cardinality_rule == 'scale':
                sorted_counts, sorted_frequencies = _get_cardinality_frequency(data)
                self._data_cardinality_scale = {
                    'num_repetitions': sorted_counts,
                    'frequency': sorted_frequencies,
                }

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

    def _sample_from_generator(self, num_samples):
        """Generate samples.

        Generate values following the regex until either the sample size is reached or
        the generator is exhausted.
        """
        samples = []
        try:
            while len(samples) < num_samples:
                samples.append(next(self.generator))
                self.generated += 1
        except (RuntimeError, StopIteration):
            pass

        return samples

    def _sample_from_template(self, num_samples, template_samples):
        """Sample num_samples values from template_samples in a cycle.

        Eg: num_samples = 5, template_samples = ['a', 'b']
        The output will be ['a', 'b', 'a', 'b', 'a']
        """
        repeats = num_samples // len(template_samples) + 1
        return np.tile(template_samples, repeats)[:num_samples].tolist()

    def _sample_match(self, num_samples):
        """Sample num_samples values following the 'match' cardinality rule."""
        samples = self._unique_regex_values[:num_samples]
        if num_samples > len(samples):
            new_samples = self._sample_from_template(num_samples - len(samples), samples)
            samples.extend(new_samples)

        return samples

    def _sample_scale_fallback(self, num_samples, template_samples):
        """Sample num_samples values, disregarding the regex, for the cardinality rule 'scale'."""
        warnings.warn(
            f"The regex for '{self.get_input_column()}' cannot generate enough samples. "
            'Additional values may not exactly follow the provided regex.'
        )
        samples = []
        fallback_samples = self._sample_fallback(num_samples, template_samples)
        while num_samples > len(samples):
            new_samples, self._remaining_samples = _sample_repetitions(
                num_samples - len(samples),
                fallback_samples.pop(0),
                self._data_cardinality_scale.copy(),
                self._remaining_samples.copy(),
            )
            samples.extend(new_samples)

        return samples

    def _sample_repetitions_from_generator(self, num_samples):
        """Sample num_samples values from the generator, or until the generator is exhausted."""
        samples = [self._remaining_samples['value']] * self._remaining_samples['repetitions']
        self._remaining_samples['repetitions'] = 0

        template_samples = []
        while num_samples > len(samples):
            try:
                value = next(self.generator)
                template_samples.append(value)
            except (RuntimeError, StopIteration):
                # If the generator is exhausted and no samples have been generated yet, reset it
                if len(template_samples) == 0:
                    self.reset_randomization()
                    continue
                else:
                    break

            new_samples, self._remaining_samples = _sample_repetitions(
                num_samples - len(samples),
                value,
                self._data_cardinality_scale.copy(),
                self._remaining_samples.copy(),
            )
            samples.extend(new_samples)

        return samples, template_samples

    def _sample_scale(self, num_samples):
        """Sample num_samples values following the 'scale' cardinality rule."""
        if self._remaining_samples['repetitions'] > num_samples:
            self._remaining_samples['repetitions'] -= num_samples
            return [self._remaining_samples['value']] * num_samples

        samples, template_samples = self._sample_repetitions_from_generator(num_samples)
        if num_samples > len(samples):
            new_samples = self._sample_scale_fallback(num_samples - len(samples), template_samples)
            samples.extend(new_samples)

        return samples

    def _sample(self, num_samples, unique_condition):
        """Sample num_samples values."""
        if num_samples <= 0:
            return []

        if hasattr(self, 'cardinality_rule'):
            if self.cardinality_rule == 'match':
                return self._sample_match(num_samples)

            if self.cardinality_rule == 'scale':
                return self._sample_scale(num_samples)

        # If there aren't enough values left in the generator, reset it
        if num_samples > self.generator_size - self.generated:
            self.reset_randomization()

        samples = self._sample_from_generator(num_samples)
        if num_samples > len(samples):
            if unique_condition:
                new_samples = self._sample_fallback(num_samples - len(samples), samples)
            else:
                new_samples = self._sample_from_template(num_samples - len(samples), samples)
            samples.extend(new_samples)

        return samples

    def _generate_unique_regexes(self):
        regex_values = []
        try:
            while len(regex_values) < self._data_cardinality:
                regex_values.append(next(self.generator))
        except (RuntimeError, StopIteration):
            fallback_samples = self._sample_fallback(
                num_samples=self._data_cardinality - len(regex_values),
                template_samples=regex_values,
            )
            regex_values.extend(fallback_samples)

        return regex_values

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
            match_cardinality = self.cardinality_rule == 'match'
            if match_cardinality and self._unique_regex_values is None:
                self._unique_regex_values = self._generate_unique_regexes()
        else:
            unique_condition = self.enforce_uniqueness
            match_cardinality = False

        num_samples = len(data) if (data is not None and len(data)) else self.data_length
        self._warn_not_enough_unique_values(num_samples, unique_condition, match_cardinality)
        samples = self._sample(num_samples, unique_condition)

        if getattr(self, 'generation_order', 'alphanumeric') == 'scrambled':
            np.random.shuffle(samples)

        return np.array(samples, dtype=object)
