"""Transformers for numerical data."""
import copy
import sys

import numpy as np
import pandas as pd
import scipy

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
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

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
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (numpy.ndarray):
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
            data (pandas.Series or numpy.ndarray):
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
            data (pandas.Series or numpy.ndarray):
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
            data (numpy.ndarray):
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
