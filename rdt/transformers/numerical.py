"""Transformers for numerical data."""
import copy

import numpy as np
import pandas as pd
import scipy

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer

EPSILON = np.finfo(np.float32).eps


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
    """

    null_transformer = None

    def __init__(self, dtype=None, nan='mean', null_column=None):
        self.nan = nan
        self.null_column = null_column
        self.dtype = dtype
        self._dtype = dtype

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self._dtype = self.dtype or data.dtype
        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        self.null_transformer.fit(data)

    def transform(self, data):
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

    def reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if np.dtype(self._dtype).kind == 'i':
            if pd.notnull(data).all():
                return data.round().astype(self._dtype)

            data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(self._dtype)
            return data

        return data.astype(self._dtype)


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

        raise TypeError('Invalid distribution: {}'.format(distribution))

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit to.
        """
        self._univariate = self._get_univariate()

        super().fit(data)
        data = super().transform(data)
        if data.ndim > 1:
            data = data[:, 0]

        self._univariate.fit(data)

    def _copula_transform(self, data):
        cdf = self._univariate.cdf(data)
        return scipy.stats.norm.ppf(cdf.clip(0 + EPSILON, 1 - EPSILON))

    def transform(self, data):
        """Transform numerical data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        transformed = super().transform(data)
        if transformed.ndim > 1:
            transformed[:, 0] = self._copula_transform(transformed[:, 0])
        else:
            transformed = self._copula_transform(transformed)

        return transformed

    def reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if data.ndim > 1:
            data[:, 0] = self._univariate.ppf(scipy.stats.norm.cdf(data[:, 0]))
        else:
            data = self._univariate.ppf(scipy.stats.norm.cdf(data))

        return super().reverse_transform(data)
