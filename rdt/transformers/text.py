"""Transformers for text data."""

import numpy as np

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer
from rdt.transformers.utils import strings_from_regex


class RegexGenerator(BaseTransformer):
    """RegexGenerator transformer.

    This transformer will drop a column and regenerate it with the previously specified
    ``regex`` format. The transformer will also be able to handle nulls and regenerate null values
    if specified.

    Args:
        regex (str):
            String representing the regex function.
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

    DETERMINISTIC_TRANSFORM = False
    DETERMINISTIC_REVERSE = False
    INPUT_SDTYPE = 'text'
    null_transformer = None

    def __init__(self, regex_format='[A-Za-z]{5}', missing_value_replacement=None,
                 model_missing_values=False):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values
        self.regex_format = regex_format

    def get_output_sdtypes(self):
        """Return the output sdtypes supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported sdtypes.
        """
        output_sdtypes = {}
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes['is_null'] = 'float'

        return self._add_prefix(output_sdtypes)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(
            self.missing_value_replacement,
            self.model_missing_values
        )
        self.null_transformer.fit(data)
        self.data_length = len(data)

    def _transform(self, data):
        """Drop the column and return ``null`` column if ``models_missing_values``.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            (numpy.ndarray or None):
                If ``self.model_missing_values`` is ``True`` then will return a ``numpy.ndarray``
                indicating which values should be ``nan``, else will return ``None``. In both
                scenarios the original column is being dropped.
        """
        if self.null_transformer and self.null_transformer.models_missing_values():
            return self.null_transformer.transform(data)[:, 1].astype(float)

        return None

    def _reverse_transform(self, data):
        """Generate new data using the provided ``regex_format``.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        generator, size = strings_from_regex(self.regex_format)
        if size > self.data_length:
            reverse_transformed = np.array([
                generator.__next__()
                for _ in range(self.data_length)
            ])

        else:
            generated_values = list(generator)
            reverse_transformed = []
            while len(reverse_transformed) < self.data_length:
                remaining = self.data_length - len(reverse_transformed)
                reverse_transformed.extend(generated_values[:remaining])

            reverse_transformed = np.array(reverse_transformed)

        if self.null_transformer.models_missing_values():
            reverse_transformed = np.column_stack((reverse_transformed, data))

        return self.null_transformer.reverse_transform(reverse_transformed)
