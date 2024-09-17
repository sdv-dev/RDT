"""Transformers for ID data."""

import logging
import warnings

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.utils import strings_from_regex

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
        regex (str):
            String representing the regex function.
        enforce_uniqueness (bool):
            Whether or not to ensure that the new generated data is all unique. If it isn't
            possible to create the requested number of rows, then an error will be raised.
            Defaults to ``False``.
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
        enforce_uniqueness=False,
        generation_order='alphanumeric',
    ):
        super().__init__()
        self.output_properties = {None: {'next_transformer': None}}
        self.enforce_uniqueness = enforce_uniqueness
        self.regex_format = regex_format
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

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.reset_randomization()
        self.data_length = len(data)

    def _transform(self, _data):
        """Drop the input column by returning ``None``."""
        return None

    def _warn_not_enough_unique_values(self, sample_size):
        """Warn the user that the regex cannot generate enough unique values.

        Args:
            sample_size (int):
                Number of samples to be generated.
        """
        warned = False
        if sample_size > self.generator_size:
            if self.enforce_uniqueness:
                warnings.warn(
                    f"The regex for '{self.get_input_column()}' can only generate "
                    f'{self.generator_size} unique values. Additional values may not exactly '
                    'follow the provided regex.'
                )
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
        if sample_size > remaining and self.enforce_uniqueness and not warned:
            warnings.warn(
                f'The regex generator is not able to generate {sample_size} new unique '
                f'values (only {max(remaining, 0)} unique values left).'
            )

    def _reverse_transform(self, data):
        """Generate new data using the provided ``regex_format``.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if data is not None and len(data):
            sample_size = len(data)
        else:
            sample_size = self.data_length

        self._warn_not_enough_unique_values(sample_size)

        remaining = self.generator_size - self.generated
        if sample_size > remaining:
            self.reset_randomization()
            remaining = self.generator_size

        generated_values = []
        while len(generated_values) < sample_size:
            try:
                generated_values.append(next(self.generator))
                self.generated += 1
            except (RuntimeError, StopIteration):
                # Can't generate more rows without collision so breaking out of loop
                break

        reverse_transformed = generated_values[:]

        if len(reverse_transformed) < sample_size:
            if self.enforce_uniqueness:
                try:
                    remaining_samples = sample_size - len(reverse_transformed)
                    start = int(generated_values[-1]) + 1
                    reverse_transformed.extend([
                        str(i) for i in range(start, start + remaining_samples)
                    ])

                except ValueError:
                    counter = 0
                    while len(reverse_transformed) < sample_size:
                        remaining_samples = sample_size - len(reverse_transformed)
                        reverse_transformed.extend([
                            f'{i}({counter})' for i in generated_values[:remaining_samples]
                        ])
                        counter += 1

            else:
                while len(reverse_transformed) < sample_size:
                    remaining_samples = sample_size - len(reverse_transformed)
                    reverse_transformed.extend(generated_values[:remaining_samples])

        if getattr(self, 'generation_order', 'alphanumeric') == 'scrambled':
            np.random.shuffle(reverse_transformed)

        return np.array(reverse_transformed, dtype=object)
