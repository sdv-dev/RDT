"""Transformers for text data."""
import warnings

import numpy as np

from rdt.errors import TransformerProcessingError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.utils import strings_from_regex


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
    """

    IS_GENERATOR = True
    INPUT_SDTYPE = 'text'

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

    def __init__(self, regex_format='[A-Za-z]{5}', enforce_uniqueness=False):
        super().__init__()
        self.output_properties = {None: {'next_transformer': None}}
        self.enforce_uniqueness = enforce_uniqueness
        self.regex_format = regex_format
        self.data_length = None
        self.generator = None
        self.generator_size = None
        self.generated = None

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

        if sample_size > self.generator_size:
            if self.enforce_uniqueness:
                raise TransformerProcessingError(
                    f'The regex is not able to generate {sample_size} unique values. '
                    f"Please use a different regex for column ('{self.get_input_column()}')."
                )

            warnings.warn(
                f"The data has {sample_size} rows but the regex for '{self.get_input_column()}' "
                f'can only create {self.generator_size} unique values. Some values in '
                f"'{self.get_input_column()}' may be repeated."
            )

        remaining = self.generator_size - self.generated
        if sample_size > self.generator_size - self.generated:
            if self.enforce_uniqueness:
                raise TransformerProcessingError(
                    f'The regex generator is not able to generate {sample_size} new unique '
                    f'values (only {remaining} unique value left). Please use '
                    "'reset_randomization' in order to restart the generator."
                )

            self.reset_randomization()
            remaining = self.generator_size

        if remaining >= sample_size:
            reverse_transformed = np.array([
                next(self.generator)
                for _ in range(sample_size)
            ], dtype=object)

            self.generated += sample_size

        else:
            self.generated = self.generator_size
            generated_values = list(self.generator)
            reverse_transformed = []
            while len(reverse_transformed) < sample_size:
                remaining_samples = sample_size - len(reverse_transformed)
                reverse_transformed.extend(generated_values[:remaining_samples])

            reverse_transformed = np.array(reverse_transformed, dtype=object)

        return reverse_transformed
