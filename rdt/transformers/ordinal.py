"""Transformers for ordinal data."""

import logging
import warnings

import numpy as np
import pandas as pd

from rdt.errors import TransformerInputError
from rdt.transformers.categorical import (
    LabelEncoder,
    UniformEncoder,
    _validate_missing_value_encoding,
)
from rdt.transformers.utils import (
    fill_nan_with_none,
)

LOGGER = logging.getLogger(__name__)


class OrderedUniformEncoder(UniformEncoder):
    """Ordered uniform encoder for categorical data.

    This class works very similarly to the ``UniformEncoder``, except that it requires the ordering
    for the labels to be provided.
    Null values are considered just another category.

    Args:
        order (list, optional):
            A list of all the unique categories for the data. The order of the list determines the
            label that each category will get. If None, orders the real data alphanumerically.
            Defaults to None.
        missing_value_encoding (str or None):
            How to encode missing values. If ``'new_category'``, missing values are encoded as
            their own category. If ``None``, missing values are not encoded and remain missing.
            Defaults to ``'new_category'``.
    """

    INPUT_SDTYPE = 'ordinal'
    SUPPORTED_SDTYPES = ['ordinal', 'categorical', 'boolean', 'id']

    def __init__(self, order=None, missing_value_encoding='new_category'):
        _validate_missing_value_encoding(missing_value_encoding)
        self.missing_value_encoding = missing_value_encoding
        self.order = order
        if order is not None:
            self.order = fill_nan_with_none(pd.Series(order))

            if not self.order.is_unique:
                error_msg = (
                    "The OrderedUniformEncoder has duplicate categories in the 'order' parameter. "
                    'Please drop the duplicates to proceed.'
                )
                raise TransformerInputError(error_msg)

        super().__init__(missing_value_encoding=missing_value_encoding)

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.get_name()
        custom_args = ['order=<CUSTOM>']
        if self.missing_value_encoding != 'new_category':
            custom_args.append(f'missing_value_encoding={repr(self.missing_value_encoding)}')

        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

    def _get_order(self, data):
        order = self.order
        if self.order is None:
            order = pd.Series(data).dropna().drop_duplicates()
            try:
                order = order.sort_values()
            except TypeError:
                warnings.warn(
                    f'The data in column `{self.get_input_column()}` contains mixed dtypes and no '
                    '`order` is provided. Defaulting to alphabetical order.'
                )
                order = order.sort_values(key=lambda x: x.astype(str))

            if pd.isna(data).any():
                order = np.append(order, [np.nan])

        if self.missing_value_encoding is None:
            order = self.order[~pd.isna(self.order)]

        return order

    def _check_unknown_categories(self, data):
        order = self._get_order(data)
        unknown = ~data.isin(order)
        if self.missing_value_encoding is None or pd.isna(order).any():
            unknown &= ~pd.isna(data)

        missing = list(data[unknown].unique())
        if len(missing) > 0:
            raise TransformerInputError(
                f"Unknown categories '{missing}'. All possible categories must be defined in the "
                "'order' parameter."
            )

    def _fit(self, data):
        """Fit the transformer to the data.

        Create all the class attributes while respecting the speicified
        order of the labels.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtypes
        order = self._get_order(data)
        if self.missing_value_encoding is None:
            data = data[~pd.isna(data)]
        else:
            data = fill_nan_with_none(data)

        self._check_unknown_categories(data)

        category_not_seen = set(order.dropna()) != set(data.dropna())
        nans_not_seen = (
            self.missing_value_encoding == 'new_category'
            and pd.isna(order).any()
            and not pd.isna(data).any()
        )
        if category_not_seen or nans_not_seen:
            unseen_categories = [x for x in order if x not in data.array]
            categories_to_print = self._get_message_unseen_categories(unseen_categories)
            LOGGER.info(
                "For column '%s', some of the provided category values were not present in the"
                ' data during fit: (%s).',
                self.get_input_column(),
                categories_to_print,
            )

            freq = data.value_counts(normalize=True, dropna=False)
            freq = 0.9 * freq
            for category in unseen_categories:
                freq.loc[category] = 0.1 / len(unseen_categories)

        else:
            freq = data.value_counts(normalize=True, dropna=False)

        nan_value = freq[np.nan] if np.nan in freq.index else None
        freq = freq.reindex(order, fill_value=nan_value).array

        self.frequencies, self.intervals = self._compute_frequencies_intervals(order, freq)

    def _transform(self, data):
        """Map the category to a continuous value."""
        if self.missing_value_encoding == 'new_category':
            data = fill_nan_with_none(data)

        self._check_unknown_categories(data)
        return super()._transform(data)


class OrderedLabelEncoder(LabelEncoder):
    """Custom label encoder for categorical data.

    This class works very similarly to the ``LabelEncoder``, except that it requires the ordering
    for the labels to be provided.

    Null values are considered just another category.

    Args:
        order (list or None, optional):
            A list of all the unique categories for the data. The order of the list determines the
            label that each category will get. If None, orders the real data alphanumerically.
            Defaults to None.
        add_noise (bool):
            Whether to generate uniform noise around the label for each category.
            Defaults to ``False``.
        missing_value_encoding (str or None):
            How to encode missing values. If ``'new_category'``, missing values are encoded as
            their own category. If ``None``, missing values are not encoded and remain missing.
            Defaults to ``'new_category'``.
    """

    INPUT_SDTYPE = 'ordinal'
    SUPPORTED_SDTYPES = ['ordinal', 'categorical', 'boolean', 'id']

    def __init__(self, order=None, add_noise=False, missing_value_encoding='new_category'):
        _validate_missing_value_encoding(missing_value_encoding)
        self.missing_value_encoding = missing_value_encoding
        self.order = None if order is None else pd.Series(order).fillna(np.nan)
        if self.order is not None and not self.order.is_unique:
            err_msg = (
                "The OrderedLabelEncoder has duplicate categories in the 'order' parameter. "
                'Please drop the duplicates to proceed.'
            )
            raise TransformerInputError(err_msg)

        super().__init__(add_noise=add_noise, missing_value_encoding=missing_value_encoding)

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.get_name()
        custom_args = []
        custom_args.append('order=<CUSTOM>')
        if self.add_noise:
            custom_args.append(f'add_noise={self.add_noise}')
        if self.missing_value_encoding != 'new_category':
            custom_args.append(f'missing_value_encoding={repr(self.missing_value_encoding)}')

        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

    def _get_order(self, data):
        order = self.order
        if self.order is None:
            order = pd.Series(data).dropna().drop_duplicates()
            try:
                order = order.sort_values()
            except TypeError:
                warnings.warn(
                    f'The data in column `{self.get_input_column()}` contains mixed dtypes and no '
                    '`order` is provided. Defaulting to alphabetical order.'
                )
                order = order.sort_values(key=lambda x: x.astype(str))

            if pd.isna(data).any():
                order = np.append(order, [np.nan])

        if self.missing_value_encoding is None:
            order = self.order[~pd.isna(self.order)]

        return order

    def _fit(self, data):
        """Fit the transformer to the data.

        Generate a unique integer representation for each category and
        store them in the ``categories_to_values`` dict and its reverse
        ``values_to_categories``.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        data = data.infer_objects()
        if self.missing_value_encoding == 'new_category':
            data = data.fillna(np.nan)

        order = self._get_order(data)

        unknown = ~data.isin(order)
        if self.missing_value_encoding is None or pd.isna(order).any():
            unknown &= ~pd.isna(data)

        missing = list(data[unknown].unique())
        if len(missing) > 0:
            raise TransformerInputError(
                f"Unknown categories '{missing}'. All possible categories must be defined in the "
                "'order' parameter."
            )

        if self.missing_value_encoding is None:
            data = data[~pd.isna(data)]

        self.values_to_categories = dict(enumerate(order))
        self.categories_to_values = {
            category: value for value, category in self.values_to_categories.items()
        }
