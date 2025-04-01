"""Transformers for categorical data."""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from rdt.errors import TransformerInputError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.utils import (
    check_nan_in_transform,
    fill_nan_with_none,
    try_convert_to_dtype,
)

LOGGER = logging.getLogger(__name__)


class UniformEncoder(BaseTransformer):
    """Transformer for categorical data.

    This transformer computes a float representative for each one of the categories
    found in the fit data, and then replaces the instances of these categories with
    the corresponding representative.

    The representatives are decided by computing the frequencies of each labels and
    then dividing the ``[0, 1]`` interval according to these frequencies.

    When the transformation is reverted, each value is assigned the category that
    corresponds to the interval it falls in.

    Null values are considered just another category.

    Args:
        order_by (str or None):
            String defining how to order the data before applying the labels. Options are
            'alphabetical', 'numerical' and ``None``. Defaults to ``None``.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean', 'id', 'text']
    frequencies = None
    intervals = None
    dtype = None

    def __init__(self, order_by=None):
        super().__init__()
        if order_by not in [None, 'alphabetical', 'numerical_value']:
            raise TransformerInputError(
                "order_by must be one of the following values: None, 'numerical_value' or "
                "'alphabetical'"
            )

        self.order_by = order_by

    def _order_categories(self, unique_data):
        nans = pd.isna(unique_data)
        if self.order_by == 'alphabetical':
            # pylint: disable=invalid-unary-operand-type
            if any(map(lambda item: not isinstance(item, str), unique_data[~nans])):  # noqa: C417
                raise TransformerInputError(
                    "The data must be of type string if order_by is 'alphabetical'."
                )
        elif self.order_by == 'numerical_value':
            if not np.issubdtype(unique_data.dtype.type, np.number):
                raise TransformerInputError(
                    "The data must be numerical if order_by is 'numerical_value'."
                )

        if self.order_by is not None:
            unique_data = np.sort(unique_data[~nans])  # pylint: disable=invalid-unary-operand-type
            if nans.any():
                unique_data = np.append(unique_data, [None])

        return unique_data

    @classmethod
    def _get_message_unseen_categories(cls, unseen_categories):
        """Message to raise when there is unseen categories.

        Args:
            unseen_categories (list): list of unseen categories

        Returns:
            message to print
        """
        categories_to_print = ', '.join(str(x) for x in unseen_categories[:3])
        if len(unseen_categories) > 3:
            categories_to_print = f'{categories_to_print}, +{len(unseen_categories) - 3} more'

        return categories_to_print

    @staticmethod
    def _compute_frequencies_intervals(categories, freq):
        """Compute the frequencies and intervals of the categories.

        Args:
            categories (list):
                List of categories.
            freq (list):
                List of frequencies.

        Returns:
            tuple[dict, dict]:
                First dict maps categories to their frequency and the
                second dict maps the categories to their intervals.
        """
        frequencies = dict(zip(categories, freq))
        shift = np.cumsum(np.hstack([0, freq]))
        shift[-1] = 1
        list_int = [[shift[i], shift[i + 1]] for i in range(len(shift) - 1)]
        intervals = dict(zip(categories, list_int))

        return frequencies, intervals

    def _fit(self, data):
        """Fit the transformer to the data.

        Compute the frequencies of each category and use them
        to map the column to a numerical one.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtypes
        data = fill_nan_with_none(data)
        labels = pd.unique(data)
        labels = self._order_categories(labels)
        freq = data.value_counts(normalize=True, dropna=False)
        nan_value = freq[np.nan] if np.nan in freq.index else None
        freq = freq.reindex(labels, fill_value=nan_value).array

        self.frequencies, self.intervals = self._compute_frequencies_intervals(labels, freq)

    def _set_fitted_parameters(self, column_name, intervals, dtype='object'):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column for this transformer.
            intervals (dict[str, tuple]):
                A dictionary mapping categories to the interval in the range [0, 1]
                it should map to.
            dtype (str, optional):
                The dtype to convert the reverse transformed data back to. Defaults to 'object'.
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        self.intervals = intervals
        self.dtype = dtype

    def _transform(self, data):
        """Map the category to a continuous value.

        This value is sampled from a uniform distribution
        with boudaries defined by the frequencies.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            pandas.Series
        """
        data_with_none = fill_nan_with_none(data)
        unseen_indexes = ~(data_with_none.isin(self.frequencies))
        if unseen_indexes.any():
            # Keep the 3 first unseen categories
            unseen_categories = list(data.loc[unseen_indexes].unique())
            categories_to_print = self._get_message_unseen_categories(unseen_categories)
            warnings.warn(
                f"The data in column '{self.get_input_column()}' contains new categories "
                f"that did not appear during 'fit' ({categories_to_print}). Assigning "
                'them random values. If you want to model new categories, '
                "please fit the data again using 'fit'.",
                category=UserWarning,
            )

            choices = list(self.frequencies.keys())
            size = unseen_indexes.size
            data_with_none[unseen_indexes] = np.random.choice(choices, size=size)

        def map_labels(label):
            return np.random.uniform(self.intervals[label][0], self.intervals[label][1])

        return data_with_none.map(map_labels).astype(float)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pandas.Series):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        data = data.clip(0, 1)
        bins = [0]
        labels = []
        nan_name = 'NaN'
        while nan_name in self.intervals.keys():
            nan_name += '_'

        for key, interval in self.intervals.items():
            bins.append(interval[1])
            if pd.isna(key):
                labels.append(nan_name)
            else:
                labels.append(key)

        result = pd.cut(data, bins=bins, labels=labels, include_lowest=True)
        if nan_name in result.cat.categories:
            result = result.cat.remove_categories(nan_name)

        result = try_convert_to_dtype(result, self.dtype)

        return result


class OrderedUniformEncoder(UniformEncoder):
    """Ordered uniform encoder for categorical data.

    This class works very similarly to the ``UniformEncoder``, except that it requires the ordering
    for the labels to be provided.
    Null values are considered just another category.

    Args:
        order (list):
            A list of all the unique categories for the data. The order of the list determines the
            label that each category will get.
    """

    def __init__(self, order):
        self.order = fill_nan_with_none(pd.Series(order))
        if not self.order.is_unique:
            error_msg = (
                "The OrderedUniformEncoder has duplicate categories in the 'order' parameter. "
                'Please drop the duplicates to proceed.'
            )
            raise TransformerInputError(error_msg)

        super().__init__()

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.get_name()
        custom_args = ['order=<CUSTOM>']
        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

    def _check_unknown_categories(self, data):
        missing = list(data[~data.isin(self.order)].unique())
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
        data = fill_nan_with_none(data)
        self._check_unknown_categories(data)

        category_not_seen = set(self.order.dropna()) != set(data.dropna())
        nans_not_seen = pd.isna(self.order).any() and not pd.isna(data).any()
        if category_not_seen or nans_not_seen:
            unseen_categories = [x for x in self.order if x not in data.array]
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
                freq[category] = 0.1 / len(unseen_categories)

        else:
            freq = data.value_counts(normalize=True, dropna=False)

        nan_value = freq[np.nan] if np.nan in freq.index else None
        freq = freq.reindex(self.order, fill_value=nan_value).array

        self.frequencies, self.intervals = self._compute_frequencies_intervals(self.order, freq)

    def _transform(self, data):
        """Map the category to a continuous value."""
        data = fill_nan_with_none(data)
        self._check_unknown_categories(data)
        return super()._transform(data)


class FrequencyEncoder(BaseTransformer):
    """Transformer for categorical data.

    This transformer computes a float representative for each one of the categories
    found in the fit data, and then replaces the instances of these categories with
    the corresponding representative.

    The representatives are decided by sorting the categorical values by their relative
    frequency, then dividing the ``[0, 1]`` interval by these relative frequencies, and
    finally assigning the middle point of each interval to the corresponding category.

    When the transformation is reverted, each value is assigned the category that
    corresponds to the interval it falls in.

    Null values are considered just another category.

    Args:
        add_noise (bool):
            Whether to generate gaussian noise around the class representative of each interval
            or just use the mean for all the replaced values. Defaults to ``False``.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean']
    mapping = None
    intervals = None
    starts = None
    means = None
    dtype = None

    def __setstate__(self, state):
        """Replace any ``null`` key by the actual ``np.nan`` instance."""
        intervals = state.get('intervals')
        if intervals:
            for key in list(intervals):
                if pd.isna(key):
                    intervals[np.nan] = intervals.pop(key)

        self.__dict__ = state

    def __init__(self, add_noise=False):
        warnings.warn(
            "The 'FrequencyEncoder' transformer will no longer be supported in future versions "
            "of the RDT library. Please use the 'UniformEncoder' transformer instead.",
            FutureWarning,
        )
        super().__init__()
        self.add_noise = add_noise
        self._is_integer = None

    @staticmethod
    def _get_intervals(data):
        """Compute intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to analyze.

        Returns:
            dict:
                intervals for each categorical value (start, end).
        """
        data = data.infer_objects().fillna(np.nan)
        frequencies = data.value_counts(dropna=False)
        augmented_frequencies = frequencies.to_frame()
        sortable_column_name = f'sortable_{frequencies.name}'
        column_name = frequencies.name or 0
        data_with_new_index = data.reset_index(drop=True)
        data_is_na = data_with_new_index.isna()

        def tie_breaker(element):
            if pd.isna(element):
                return data_is_na.loc[data_is_na == 1].index[0]

            return data_with_new_index.loc[data_with_new_index == element].index[0]

        augmented_frequencies[sortable_column_name] = frequencies.index.map(tie_breaker)
        augmented_frequencies = augmented_frequencies.sort_values(
            [column_name, sortable_column_name], ascending=[False, True]
        )
        sorted_frequencies = augmented_frequencies[column_name]

        start = 0
        end = 0
        elements = len(data)

        intervals = {}
        means = []
        starts = []
        for value, frequency in sorted_frequencies.items():
            prob = frequency / elements
            end = start + prob
            mean = start + prob / 2
            std = prob / 6
            if pd.isna(value):
                value = np.nan

            intervals[value] = (start, end, mean, std)
            means.append(mean)
            starts.append((value, start))
            start = end

        means = pd.Series(means, index=list(frequencies.keys()))
        starts = pd.DataFrame(starts, columns=['category', 'start']).set_index('start')

        return intervals, means, starts

    def _fit(self, data):
        """Fit the transformer to the data.

        Compute the intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        self.intervals, self.means, self.starts = self._get_intervals(data)

    @staticmethod
    def _clip_noised_transform(result, start, end):
        """Clip transformed values.

        Used to ensure the noise added to transformed values doesn't make it
        go out of the bounds of a given category.

        The upper bound must be slightly lower than ``end``
        so it doesn't get treated as the next category.
        """
        return np.clip(result, start, end - 1e-9)

    def _transform_by_category(self, data):
        """Transform the data by iterating over the different categories."""
        result = np.empty(shape=(len(data),), dtype=float)

        # loop over categories
        for category, values in self.intervals.items():
            start, end, mean, std = values
            if category is np.nan:
                mask = data.isna()
            else:
                mask = data.to_numpy() == category

            if self.add_noise:
                result[mask] = norm.rvs(
                    mean,
                    std,
                    size=mask.sum(),
                    random_state=self.random_states['transform'],
                )
                result[mask] = self._clip_noised_transform(result[mask], start, end)
            else:
                result[mask] = mean

        return result

    def _get_value(self, category):
        """Get the value that represents this category."""
        if pd.isna(category):
            category = np.nan

        start, end, mean, std = self.intervals[category]

        if self.add_noise:
            result = norm.rvs(mean, std, random_state=self.random_states['transform'])
            return self._clip_noised_transform(result, start, end)

        return mean

    def _transform_by_row(self, data):
        """Transform the data row by row."""
        data = data.infer_objects().fillna(np.nan).apply(self._get_value).to_numpy()

        return data

    def _transform(self, data):
        """Transform the categorical values to float representatives.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        fit_categories = pd.Series(self.intervals.keys())
        has_nan = pd.isna(fit_categories).any()
        unseen_indexes = ~(data.isin(fit_categories) | (pd.isna(data) & has_nan))
        if unseen_indexes.any():
            # Select only the first 5 unseen categories to avoid flooding the console.
            unseen_categories = set(data[unseen_indexes][:5])
            warnings.warn(
                f'The data contains {unseen_indexes.sum()} new categories that were not '
                f'seen in the original data (examples: {unseen_categories}). Assigning '
                'them random values. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

        data[unseen_indexes] = np.random.choice(fit_categories, size=unseen_indexes.size)
        if len(self.means) < len(data):
            return self._transform_by_category(data)

        return self._transform_by_row(data)

    def _reverse_transform_by_category(self, data):
        """Reverse transform the data by iterating over all the categories."""
        result = np.empty(shape=(len(data),), dtype=self.dtype)

        # loop over categories
        for category, values in self.intervals.items():
            start = values[0]
            mask = start <= data.to_numpy()
            result[mask] = category

        return pd.Series(result, index=data.index, dtype=self.dtype)

    def _get_category_from_start(self, value):
        lower = self.starts.loc[:value]
        return lower.iloc[-1].category

    def _reverse_transform_by_row(self, data):
        """Reverse transform the data by iterating over each row."""
        return data.apply(self._get_category_from_start).astype(self.dtype)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        data = data.clip(0, 1)
        num_rows = len(data)
        num_categories = len(self.means)

        if num_rows > num_categories:
            return self._reverse_transform_by_category(data)

        # loop over rows
        return self._reverse_transform_by_row(data)


class OneHotEncoder(BaseTransformer):
    """OneHotEncoding for categorical data.

    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.

    Null values are considered just another category.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean']
    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None
    dtype = None

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.

        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def _fit(self, data):
        """Fit the transformer to the data.

        Get the pandas `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        data = self._prepare_data(data)

        null = pd.isna(data).to_numpy()
        self._uniques = list(pd.unique(data[~null]))
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not np.issubdtype(data.dtype.type, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

        self.output_properties = {
            f'value{i}': {'sdtype': 'float', 'next_transformer': None}
            for i in range(len(self.dummies))
        }

    def _transform_helper(self, data):
        if self._dummy_encoded:
            coder = self._indexer
            codes = pd.Categorical(data, categories=self._uniques).codes
        else:
            coder = self._uniques
            codes = data

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[pd.isna(data)] = 1
            array = np.append(array, null, axis=1)

        return array

    def _transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._prepare_data(data)
        unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
        unseen_categories = unique_data - {np.nan if pd.isna(x) else x for x in self.dummies}
        if unseen_categories:
            # Select only the first 5 unseen categories to avoid flooding the console.
            examples_unseen_categories = set(list(unseen_categories)[:5])
            warnings.warn(
                f'The data contains {len(unseen_categories)} new categories that were not '
                f'seen in the original data (examples: {examples_unseen_categories}). Creating '
                'a vector of all 0s. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

        return self._transform_helper(data)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        result = pd.Series(indices).map(self.dummies.__getitem__)
        result = try_convert_to_dtype(result, self.dtype)

        return result


class LabelEncoder(BaseTransformer):
    """LabelEncoding for categorical data.

    This transformer generates a unique integer representation for each category
    and simply replaces each category with its integer value.

    Null values are considered just another category.

    Attributes:
        values_to_categories (dict):
            Dictionary that maps each integer value for its category.
        categories_to_values (dict):
            Dictionary that maps each category with the corresponding
            integer value.

    Args:
        add_noise (bool):
            Whether to generate uniform noise around the label for each category.
            Defaults to ``False``.
        order_by (None or str):
            A string defining how to order the categories before assigning them labels. Defaults to
            ``None``. Options include:
            - ``'numerical_value'``: Order the categories by numerical value.
            - ``'alphabetical'``: Order the categories alphabetically.
            - ``None``: Use the order that the categories appear in when fitting.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean', 'id', 'text']
    values_to_categories = None
    categories_to_values = None
    dtype = 'O'

    def __init__(self, add_noise=False, order_by=None):
        super().__init__()
        self.add_noise = add_noise
        if order_by not in [None, 'alphabetical', 'numerical_value']:
            raise TransformerInputError(
                "order_by must be one of the following values: None, 'numerical_value' or "
                "'alphabetical'"
            )

        self.order_by = order_by

    def _order_categories(self, unique_data):
        if self.order_by == 'alphabetical':
            if unique_data.dtype.type not in [np.str_, np.object_]:
                raise TransformerInputError(
                    "The data must be of type string if order_by is 'alphabetical'."
                )

        elif self.order_by == 'numerical_value':
            if not np.issubdtype(unique_data.dtype.type, np.number):
                raise TransformerInputError(
                    "The data must be numerical if order_by is 'numerical_value'."
                )

        if self.order_by is not None:
            nans = pd.isna(unique_data)
            unique_data = np.sort(unique_data[~nans])  # pylint: disable=E1130
            if nans.any():
                unique_data = np.append(unique_data, [np.nan])

        return unique_data

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
        unique_data = pd.unique(data.infer_objects().fillna(np.nan))
        unique_data = self._order_categories(unique_data)
        self.values_to_categories = dict(enumerate(unique_data))
        self.categories_to_values = {
            category: value for value, category in self.values_to_categories.items()
        }

    def _transform(self, data):
        """Replace each category with its corresponding integer value.

        If a category has not been seen before, a random value is assigned.

        If ``add_noise`` is True, the integer values will be replaced by a
        random number between the value and the value + 1.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            pd.Series
        """
        mapped = data.infer_objects().fillna(np.nan).map(self.categories_to_values)
        is_null = mapped.isna()
        if is_null.any():
            # Select only the first 5 unseen categories to avoid flooding the console.
            unseen_categories = set(data[is_null][:5])
            warnings.warn(
                f'The data contains {is_null.sum()} new categories that were not '
                f'seen in the original data (examples: {unseen_categories}). Assigning '
                'them random values. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

        mapped[is_null] = np.random.randint(len(self.categories_to_values), size=is_null.sum())

        if self.add_noise:
            mapped = mapped.astype(float)
            mapped = np.random.uniform(mapped, mapped + 1)

        return mapped

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        if self.add_noise:
            data = np.floor(data)

        data = data.clip(min(self.values_to_categories), max(self.values_to_categories))
        data = data.round().map(self.values_to_categories)
        data = try_convert_to_dtype(data, self.dtype)

        return data


class OrderedLabelEncoder(LabelEncoder):
    """Custom label encoder for categorical data.

    This class works very similarly to the ``LabelEncoder``, except that it requires the ordering
    for the labels to be provided.

    Null values are considered just another category.

    Args:
        order (list):
            A list of all the unique categories for the data. The order of the list determines the
            label that each category will get.
        add_noise (bool):
            Whether to generate uniform noise around the label for each category.
            Defaults to ``False``.
    """

    def __init__(self, order, add_noise=False):
        self.order = pd.Series(order).fillna(np.nan)
        if not self.order.is_unique:
            err_msg = (
                "The OrderedLabelEncoder has duplicate categories in the 'order' parameter. "
                'Please drop the duplicates to proceed.'
            )
            raise TransformerInputError(err_msg)

        super().__init__(add_noise=add_noise)

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

        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

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
        data = data.infer_objects().fillna(np.nan)

        missing = list(data[~data.isin(self.order)].unique())
        if len(missing) > 0:
            raise TransformerInputError(
                f"Unknown categories '{missing}'. All possible categories must be defined in the "
                "'order' parameter."
            )

        self.values_to_categories = dict(enumerate(self.order))
        self.categories_to_values = {
            category: value for value, category in self.values_to_categories.items()
        }


class CustomLabelEncoder(OrderedLabelEncoder):
    """Deprecated class name for ``OrderedLabelEncoder``.

    Class to ensure backwards compatibility with previous versions of RDT.
    """

    def __init__(self, order, add_noise=False):
        warnings.warn(
            "The 'CustomLabelEncoder' is renamed to 'OrderedLabelEncoder'. Please update the"
            'name to ensure compatibility with future versions of RDT.',
            FutureWarning,
        )
        super().__init__(order, add_noise)
