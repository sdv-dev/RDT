"""Transformers for numerical data."""
import copy
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy
from sklearn.mixture import BayesianGaussianMixture

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer

EPSILON = np.finfo(np.float32).eps
MAX_DECIMALS = sys.float_info.dig - 1


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent.
    Non null float values are not modified.

    Null values are replaced using a ``NullTransformer``.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the
            transformation. If not provided, the dtype of the fit data will be used.
            Defaults to ``None``.
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        rounding (int, str or None):
            Define rounding scheme for data. If set to an int, values will be rounded
            to that number of decimal places. If ``None``, values will not be rounded.
            If set to ``'auto'``, the transformer will round to the maximum number of
            decimal places detected in the fitted data.
        min_value (int, str or None):
            Indicate whether or not to set a minimum value for the data. If an integer is given,
            reverse transformed data will be greater than or equal to it. If the string ``'auto'``
            is given, the minimum will be the minimum value seen in the fitted data. If ``None``
            is given, there won't be a minimum.
        max_value (int, str or None):
            Indicate whether or not to set a maximum value for the data. If an integer is given,
            reverse transformed data will be less than or equal to it. If the string ``'auto'``
            is given, the maximum will be the maximum value seen in the fitted data. If ``None``
            is given, there won't be a maximum.
    """

    INPUT_TYPE = 'numerical'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    null_transformer = None
    nan = None
    _dtype = None
    _rounding_digits = None
    _min_value = None
    _max_value = None

    def __init__(self, dtype=None, nan='mean', null_column=None, rounding=None,
                 min_value=None, max_value=None):
        self.nan = nan
        self.null_column = null_column
        self.dtype = dtype
        self.rounding = rounding
        self.min_value = min_value
        self.max_value = max_value

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        output_types = {
            'value': 'float',
        }
        if self.null_transformer and self.null_transformer.creates_null_column():
            output_types['is_null'] = 'float'

        return self._add_prefix(output_types)

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        if self.null_transformer and not self.null_transformer.creates_null_column():
            return False

        return self.COMPOSITION_IS_IDENTITY

    @staticmethod
    def _learn_rounding_digits(data):
        # check if data has any decimals
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]
        if ((roundable_data % 1) != 0).any():
            if not (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
                return None

            for decimal in range(MAX_DECIMALS + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        elif len(roundable_data) > 0:
            maximum = max(abs(roundable_data))
            start = int(np.log10(maximum)) if maximum != 0 else 0
            for decimal in range(-start, 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        return None

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.DataFrame or pandas.Series):
                Data to fit.
        """
        self._dtype = self.dtype or data.dtype
        self._min_value = data.min() if self.min_value == 'auto' else self.min_value
        self._max_value = data.max() if self.max_value == 'auto' else self.max_value

        if self.rounding == 'auto':
            self._rounding_digits = self._learn_rounding_digits(data)
        elif isinstance(self.rounding, int):
            self._rounding_digits = self.rounding

        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        self.null_transformer.fit(data)

    def _transform(self, data):
        """Transform numerical data.

        Integer values are replaced by their float equivalent. Non null float values
        are left unmodified.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if self._min_value is not None or self._max_value is not None:
            if len(data.shape) > 1:
                data[:, 0] = data[:, 0].clip(self._min_value, self._max_value)
            else:
                data = data.clip(self._min_value, self._max_value)

        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self._rounding_digits is not None or is_integer:
            data = data.round(self._rounding_digits or 0)

        if pd.isna(data).any() and is_integer:
            return data

        return data.astype(self._dtype)


class NumericalRoundedBoundedTransformer(NumericalTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent, bounded by the fitted
    data (the minimum and maximum values seen while fitting). It will also round all values to
    the maximum number of decimal places detected in the fitted data.

    Non null float values are not modified.

    This class behaves exactly as the ``NumericalTransformer`` with ``min_value='auto'``,
    ``max_value='auto'`` and ``rounding='auto'``.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the
            transformation. If not provided, the dtype of the fit data will be used.
            Defaults to ``None``.
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
    """

    def __init__(self, dtype=None, nan='mean', null_column=None):
        super().__init__(dtype=dtype, nan=nan, null_column=null_column, min_value='auto',
                         max_value='auto', rounding='auto')


class NumericalBoundedTransformer(NumericalTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent, bounded by the fitted
    data (the minimum and maximum values seen while fitting).

    Non null float values are not modified.

    This class behaves exactly as the ``NumericalTransformer`` with ``min_value='auto'``,
    ``max_value='auto'`` and ``rounding=None``.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the
            transformation. If not provided, the dtype of the fit data will be used.
            Defaults to ``None``.
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
    """

    def __init__(self, dtype=None, nan='mean', null_column=None):
        super().__init__(dtype=dtype, nan=nan, null_column=null_column, min_value='auto',
                         max_value='auto', rounding=None)


class NumericalRoundedTransformer(NumericalTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent, rounding all values to
    the maximum number of decimal places detected in the fitted data.

    Non null float values are not modified.

    This class behaves exactly as the ``NumericalTransformer`` with ``min_value=None``,
    ``max_value=None`` and ``rounding='auto'``.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the
            transformation. If not provided, the dtype of the fit data will be used.
            Defaults to ``None``.
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
    """

    def __init__(self, dtype=None, nan='mean', null_column=None):
        super().__init__(dtype=dtype, nan=nan, null_column=null_column, min_value=None,
                         max_value=None, rounding='auto')


class GaussianCopulaTransformer(NumericalTransformer):
    r"""Transformer for numerical data based on copulas transformation.

    Transformation consists on bringing the input data to a standard normal space
    by using a combination of *cdf* and *inverse cdf* transformations:

    Given a variable :math:`x`:

    - Find the best possible marginal or use user specified one, :math:`P(x)`.
    - do :math:`u = \phi (x)` where :math:`\phi` is cumulative density function,
      given :math:`P(x)`.
    - do :math:`z = \phi_{N(0,1)}^{-1}(u)`, where :math:`\phi_{N(0,1)}^{-1}` is
      the *inverse cdf* of a *standard normal* distribution.

    The reverse transform will do the inverse of the steps above and go from :math:`z`
    to :math:`u` and then to :math:`x`.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the
            transformation. If not provided, the dtype of the fit data will be used.
            Defaults to ``None``.
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use. Defaults to ``parametric``. To choose from:

                * ``univariate``: Let ``copulas`` select the optimal univariate distribution.
                  This may result in non-parametric models being used.
                * ``parametric``: Let ``copulas`` select the optimal univariate distribution,
                  but restrict the selection to parametric distributions only.
                * ``bounded``: Let ``copulas`` select the optimal univariate distribution,
                  but restrict the selection to bounded distributions only.
                  This may result in non-parametric models being used.
                * ``semi_bounded``: Let ``copulas`` select the optimal univariate distribution,
                  but restrict the selection to semi-bounded distributions only.
                  This may result in non-parametric models being used.
                * ``parametric_bounded``: Let ``copulas`` select the optimal univariate
                  distribution, but restrict the selection to parametric and bounded distributions
                  only.
                * ``parametric_semi_bounded``: Let ``copulas`` select the optimal univariate
                  distribution, but restrict the selection to parametric and semi-bounded
                  distributions only.
                * ``gaussian``: Use a Gaussian distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``beta``: Use a Beta distribution.
                * ``student_t``: Use a Student T distribution.
                * ``gussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
                * ``truncated_gaussian``: Use a Truncated Gaussian distribution.
    """

    _univariate = None
    COMPOSITION_IS_IDENTITY = False

    def __init__(self, dtype=None, nan='mean', null_column=None, distribution='parametric'):
        super().__init__(dtype=dtype, nan=nan, null_column=null_column)
        self._distributions = self._get_distributions()

        if isinstance(distribution, str):
            distribution = self._distributions[distribution]

        self._distribution = distribution

    @staticmethod
    def _get_distributions():
        try:
            from copulas import univariate  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            error.msg += (
                '\n\nIt seems like `copulas` is not installed.\n'
                'Please install it using:\n\n    pip install rdt[copulas]'
            )
            raise

        return {
            'univariate': univariate.Univariate,
            'parametric': (
                univariate.Univariate, {
                    'parametric': univariate.ParametricType.PARAMETRIC,
                },
            ),
            'bounded': (
                univariate.Univariate,
                {
                    'bounded': univariate.BoundedType.BOUNDED,
                },
            ),
            'semi_bounded': (
                univariate.Univariate,
                {
                    'bounded': univariate.BoundedType.SEMI_BOUNDED,
                },
            ),
            'parametric_bounded': (
                univariate.Univariate,
                {
                    'parametric': univariate.ParametricType.PARAMETRIC,
                    'bounded': univariate.BoundedType.BOUNDED,
                },
            ),
            'parametric_semi_bounded': (
                univariate.Univariate,
                {
                    'parametric': univariate.ParametricType.PARAMETRIC,
                    'bounded': univariate.BoundedType.SEMI_BOUNDED,
                },
            ),
            'gaussian': univariate.GaussianUnivariate,
            'gamma': univariate.GammaUnivariate,
            'beta': univariate.BetaUnivariate,
            'student_t': univariate.StudentTUnivariate,
            'gaussian_kde': univariate.GaussianKDE,
            'truncated_gaussian': univariate.TruncatedGaussian,
        }

    def _get_univariate(self):
        distribution = self._distribution
        if isinstance(distribution, self._distributions['univariate']):
            return copy.deepcopy(distribution)
        if isinstance(distribution, tuple):
            return distribution[0](**distribution[1])
        if isinstance(distribution, type) and \
           issubclass(distribution, self._distributions['univariate']):
            return distribution()

        raise TypeError(f'Invalid distribution: {distribution}')

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self._univariate = self._get_univariate()

        super()._fit(data)
        data = super()._transform(data)
        if data.ndim > 1:
            data = data[:, 0]

        self._univariate.fit(data)

    def _copula_transform(self, data):
        cdf = self._univariate.cdf(data)
        return scipy.stats.norm.ppf(cdf.clip(0 + EPSILON, 1 - EPSILON))

    def _transform(self, data):
        """Transform numerical data.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        transformed = super()._transform(data)
        if transformed.ndim > 1:
            transformed[:, 0] = self._copula_transform(transformed[:, 0])
        else:
            transformed = self._copula_transform(transformed)

        return transformed

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim > 1:
            data[:, 0] = self._univariate.ppf(scipy.stats.norm.cdf(data[:, 0]))
        else:
            data = self._univariate.ppf(scipy.stats.norm.cdf(data))

        return super()._reverse_transform(data)


SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type',
        'transform', 'transform_aux',
        'output_info', 'output_dimensions'
    ]
)


class BayesGMMTransformer(NumericalTransformer):
    """Bayesian GMM transformer."""

    DETERMINISTIC_TRANSFORM = False
    DETERMINISTIC_REVERSE = False
    COMPOSITION_IS_IDENTITY = False

    def __init__(self, dtype=None, nan='mean', null_column=None, max_clusters=10,
                 weight_threshold=0.005):
        super().__init__(dtype=dtype, nan=nan, null_column=null_column)
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self.output_info_list = None
        self.output_dimensions = None
        self.dataframe = None
        self._column_raw_dtypes = None
        self._column_transform_info_list = None

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        output_types = {
            'continuous': 'float',
            'discrete': 'categorical'
        }
        if self.null_transformer and self.null_transformer.creates_null_column():
            output_types['is_null'] = 'float'

        return self._add_prefix(output_types)

    def _fit_continuous(self, column_name, raw_column_data):
        """Train Bayesian GMM for continuous column."""
        bgm_transformer = BayesianGaussianMixture(
            n_components=self._max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )

        bgm_transformer.fit(raw_column_data.reshape(-1, 1))
        valid_component_indicator = bgm_transformer.weights_ > self._weight_threshold
        num_components = valid_component_indicator.sum()

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=bgm_transformer,
            transform_aux=valid_component_indicator,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        data = pd.DataFrame(data)

        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True
        self._column_raw_dtypes = data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in data.columns:
            raw_column_data = data[column_name].to_numpy()
            column_transform_info = self._fit_continuous(column_name, raw_column_data)
            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, raw_column_data):
        bgm_transformer = column_transform_info.transform

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()

        means = bgm_transformer.means_.reshape((1, self._max_clusters))
        stds = np.sqrt(bgm_transformer.covariances_).reshape((1, self._max_clusters))
        normalized_values = ((raw_column_data - means) / (4 * stds))[:, valid_component_indicator]
        component_probs = bgm_transformer.predict_proba(raw_column_data)
        component_probs = component_probs[:, valid_component_indicator]

        selected_component = np.zeros(len(raw_column_data), dtype='int')
        for i in range(len(raw_column_data)):
            component_porb_t = component_probs[i] + 1e-6
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_component[i] = np.random.choice(np.arange(num_components), p=component_porb_t)

        aranged = np.arange(len(raw_column_data))
        selected_normalized_value = normalized_values[aranged, selected_component].reshape([-1, 1])
        selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[np.arange(len(raw_column_data)), selected_component] = 1

        return [selected_normalized_value, selected_component_onehot]

    def _transform(self, data):
        """Transform numerical data.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = pd.DataFrame(data)
        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = data[[column_transform_info.column_name]].to_numpy()
            if column_transform_info.column_type == 'continuous':
                column_data_list += self._transform_continuous(column_transform_info, column_data)

        transformed = np.concatenate(column_data_list, axis=1).astype(float)
        normalized, one_hot = transformed[:, 0], transformed[:, 1:]
        one_hot_as_label = one_hot.argmax(axis=1)
        return pd.DataFrame({
            'continuous': normalized,
            'discrete': one_hot_as_label
        })

    def _reverse_transform_continuous(self, column_transform_info, column_data, sigmas, start):
        bgm_transformer = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[start]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = bgm_transformer.means_.reshape([-1])
        stds = np.sqrt(bgm_transformer.covariances_).reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _reverse_transform(self, data, sigmas=None):
        """Convert data back into the original format.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = pd.DataFrame(data)
        one_hot = np.zeros(shape=(data.shape[0], self.output_dimensions-1))
        discrete_column = data[self.output_columns[1]].tolist()
        one_hot[np.arange(data.shape[0]), discrete_column] = 1.0
        data = np.concatenate([data[self.output_columns[0]][:, None], one_hot], axis=1)

        start = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, start:start + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._reverse_transform_continuous(
                    column_transform_info, column_data, sigmas, start)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            start += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))

        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data
