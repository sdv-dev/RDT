"""Transformers for text data."""
import warnings

import numpy as np

from rdt.errors import Error
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
            possible to create the requested number of rows, then an ``Error`` will be raised.
            Defaults to ``False``.
    """

    DETERMINISTIC_TRANSFORM = False
    DETERMINISTIC_REVERSE = False
    IS_GENERATOR = True
    INPUT_SDTYPE = 'text'
    OUTPUT_SDTYPES = {}

    def __init__(self, regex_format='[A-Za-z]{5}', enforce_uniqueness=False):
        self.enforce_uniqueness = enforce_uniqueness
        self.regex_format = regex_format
        self.data_length = None

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
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

        generator, size = strings_from_regex(self.regex_format)

        if sample_size > size:
            if self.enforce_uniqueness:
                raise Error(
                    f'The regex is not able to generate {sample_size} unique values. '
                    f"Please use a different regex for column ('{self.get_input_column()}')."
                )

            warnings.warn(
                f"The data has {sample_size} rows but the regex for '{self.get_input_column()}' "
                f'can only create {size} unique values. Some values in '
                f"'{self.get_input_column()}' may be repeated."
            )

        if size > sample_size:
            reverse_transformed = np.array([
                next(generator)
                for _ in range(sample_size)
            ], dtype=object)

        else:
            generated_values = list(generator)
            reverse_transformed = []
            while len(reverse_transformed) < sample_size:
                remaining = sample_size - len(reverse_transformed)
                reverse_transformed.extend(generated_values[:remaining])

            reverse_transformed = np.array(reverse_transformed, dtype=object)

        return reverse_transformed
