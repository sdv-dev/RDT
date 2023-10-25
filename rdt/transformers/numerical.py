"""Transformers for numerical data."""
import copy
import logging
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from sklearn.mixture import BayesianGaussianMixture

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer

LOGGER = logging.getLogger(__name__)

EPSILON = np.finfo(np.float32).eps
MAX_DECIMALS = sys.float_info.dig - 1
INTEGER_BOUNDS = {
    'Int8': (-2**7, 2**7 - 1),
    'Int16': (-2**15, 2**15 - 1),
    'Int32': (-2**31, 2**31 - 1),
    'Int64': (-2**63, 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}


class FloatFormatter(BaseTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent.
    Non null float values are not modified.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'``
            are given, replace them with the corresponding aggregation and if ``'random'``
            replace each null value with a random value in the data range. Defaults to ``mean``.
         model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        computer_representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
    """

    INPUT_SDTYPE = 'numerical'
    null_transformer = None
    missing_value_replacement = None
    _dtype = None
    _rounding_digits = None
    _min_value = None
    _max_value = None

    def __init__(self, missing_value_replacement='mean', model_missing_values=None,
                 learn_rounding_scheme=False, enforce_min_max_values=False,
                 computer_representation='Float', missing_value_generation='random'):
        super().__init__()
        self.missing_value_replacement = missing_value_replacement
        self._set_missing_value_generation(missing_value_generation)
        self._set_missing_value_replacement('mean', missing_value_replacement)
        if model_missing_values is not None:
            self._set_model_missing_values(model_missing_values)

        self.learn_rounding_scheme = learn_rounding_scheme
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

    @staticmethod
    def _learn_rounding_digits(data):
        # check if data has any decimals
        name = data.name
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]

        # Doesn't contain numbers
        if len(roundable_data) == 0:
            return None

        # Doesn't contain decimal digits
        if ((roundable_data % 1) == 0).all():
            return 0

        # Try to round to fewer digits
        if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
            for decimal in range(MAX_DECIMALS + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        # Can't round, not equal after MAX_DECIMALS digits of precision
        LOGGER.info("No rounding scheme detected for column '%s'. Data will not be rounded.", name)
        return None

    def _raise_out_of_bounds_error(self, value, name, bound_type, min_bound, max_bound):
        raise ValueError(
            f"The {bound_type} value in column '{name}' is {value}."
            f" All values represented by '{self.computer_representation}'"
            f' must be in the range [{min_bound}, {max_bound}].'
        )

    def _validate_values_within_bounds(self, data):
        if self.computer_representation != 'Float':
            fractions = data[~data.isna() & data % 1 != 0]
            if not fractions.empty:
                raise ValueError(
                    f"The column '{data.name}' contains float values {fractions.tolist()}. "
                    f"All values represented by '{self.computer_representation}' must be integers."
                )

            min_value = data.min()
            max_value = data.max()
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            if min_value < min_bound:
                self._raise_out_of_bounds_error(
                    min_value, data.name, 'minimum', min_bound, max_bound)

            if max_value > max_bound:
                self._raise_out_of_bounds_error(
                    max_value, data.name, 'maximum', min_bound, max_bound)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit.
        """
        self._validate_values_within_bounds(data)
        self._dtype = data.dtype

        if self.enforce_min_max_values:
            self._min_value = data.min()
            self._max_value = data.max()

        if self.learn_rounding_scheme:
            self._rounding_digits = self._learn_rounding_digits(data)

        self.null_transformer = NullTransformer(
            self.missing_value_replacement,
            self.missing_value_generation
        )
        self.null_transformer.fit(data)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {'sdtype': 'float', 'next_transformer': None}

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
        self._validate_values_within_bounds(data)
        data = data.astype(np.float64)
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

        data = self.null_transformer.reverse_transform(data)
        if self.enforce_min_max_values:
            data = data.clip(self._min_value, self._max_value)
        elif self.computer_representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            data = data.clip(min_bound, max_bound)

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.learn_rounding_scheme and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        elif is_integer:
            data = data.round(0)

        if pd.isna(data).any() and is_integer:
            return data

        return data.astype(self._dtype)


class GaussianNormalizer(FloatFormatter):
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
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        distribution (copulas.univariate.Univariate or str):
            Copulas univariate distribution to use. Defaults to ``truncated_gaussian``.
            Options include:

                * ``norm``: Use a Gaussian distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``beta``: Use a Beta distribution.
                * ``t``: Use a Student T distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
                * ``truncnorm``: Use a Truncated Gaussian distribution.
                # ``uniform``: Use a UniformUnivariate distribution.

        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
    """

    _univariate = None
    _DEPRECATED_DISTRIBUTIONS_MAPPING = {
        'gaussian': 'norm',
        'student_t': 't',
        'truncated_gaussian': 'truncnorm'
    }

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
            'norm': univariate.GaussianUnivariate,
            'gamma': univariate.GammaUnivariate,
            'beta': univariate.BetaUnivariate,
            't': univariate.StudentTUnivariate,
            'gaussian_kde': univariate.GaussianKDE,
            'truncnorm': univariate.TruncatedGaussian,
            'uniform': univariate.UniformUnivariate,
        }

    def __init__(self, model_missing_values=None, learn_rounding_scheme=False,
                 enforce_min_max_values=False, distribution='truncated_gaussian',
                 missing_value_generation='random'):

        # Using missing_value_replacement='mean' as the default instead of random
        # as this may lead to different outcomes in certain synthesizers
        # affecting the synthesizers directly and this is out of scope for now.
        super().__init__(
            model_missing_values=model_missing_values,
            missing_value_generation=missing_value_generation,
            missing_value_replacement='mean',
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values
        )

        self._distributions = self._get_distributions()
        if isinstance(distribution, str):
            if distribution in {'gaussian', 'student_t', 'truncated_gaussian'}:
                warnings.warn(
                    f"Future versions of RDT will not support '{distribution}' as an option. "
                    f"Please use '{self._DEPRECATED_DISTRIBUTIONS_MAPPING[distribution]}' "
                    'instead.',
                    FutureWarning
                )
                distribution = self._DEPRECATED_DISTRIBUTIONS_MAPPING[distribution]

            distribution = self._distributions[distribution]

        self._distribution = distribution

    def _get_univariate(self):
        distribution = self._distribution
        if any(isinstance(distribution, dist) for dist in self._distributions.values()):
            return copy.deepcopy(distribution)
        if isinstance(distribution, tuple):
            return distribution[0](**distribution[1])
        if isinstance(distribution, type) and distribution in self._distributions.values():
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

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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


class ClusterBasedNormalizer(FloatFormatter):
    """Transformer for numerical data using a Bayesian Gaussian Mixture Model.

    This transformation takes a numerical value and transforms it using a Bayesian GMM
    model. It generates two outputs, a discrete value which indicates the selected
    'component' of the GMM and a continuous value which represents the normalized value
    based on the mean and std of the selected component.

    Args:
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        max_clusters (int):
            The maximum number of mixture components. Depending on the data, the model may select
            fewer components (based on the ``weight_threshold``).
            Defaults to 10.
        weight_threshold (int, float):
            The minimum value a component weight can take to be considered a valid component.
            ``weights_`` under this value will be ignored.
            Defaults to 0.005.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.

    Attributes:
        _bgm_transformer:
            An instance of sklearn`s ``BayesianGaussianMixture`` class.
        valid_component_indicator:
            An array indicating the valid components. If the weight of a component is greater
            than the ``weight_threshold``, it's indicated with True, otherwise it's set to False.
    """

    STD_MULTIPLIER = 4
    _bgm_transformer = None
    valid_component_indicator = None

    def __init__(self, model_missing_values=None, learn_rounding_scheme=False,
                 enforce_min_max_values=False, max_clusters=10, weight_threshold=0.005,
                 missing_value_generation='random'):

        # Using missing_value_replacement='mean' as the default instead of random
        # as this may lead to different outcomes in certain synthesizers
        # affecting the synthesizers directly and this is out of scope for now.
        super().__init__(
            model_missing_values=model_missing_values,
            missing_value_generation=missing_value_generation,
            missing_value_replacement='mean',
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values
        )
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.output_properties = {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None},
        }

    def _get_current_random_seed(self):
        if self.random_states:
            return self.random_states['fit'].get_state()[1][0]

        return 0

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self._bgm_transformer = BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            random_state=self._get_current_random_seed()
        )

        super()._fit(data)
        data = super()._transform(data)
        if data.ndim > 1:
            data = data[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm_transformer.fit(data.reshape(-1, 1))

        self.valid_component_indicator = self._bgm_transformer.weights_ > self.weight_threshold

    def _transform(self, data):
        """Transform the numerical data.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray.
        """
        data = super()._transform(data)
        if data.ndim > 1:
            data, model_missing_values = data[:, 0], data[:, 1]

        data = data.reshape((len(data), 1))
        means = self._bgm_transformer.means_.reshape((1, self.max_clusters))
        means = means[:, self.valid_component_indicator]
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape((1, self.max_clusters))
        stds = stds[:, self.valid_component_indicator]

        # Multiply stds by 4 so that a value will be in the range [-1,1] with 99.99% probability
        normalized_values = (data - means) / (self.STD_MULTIPLIER * stds)
        component_probs = self._bgm_transformer.predict_proba(data)
        component_probs = component_probs[:, self.valid_component_indicator]

        selected_component = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            component_prob_t = component_probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(self.valid_component_indicator.sum()),
                p=component_prob_t
            )

        aranged = np.arange(len(data))
        normalized = normalized_values[aranged, selected_component].reshape([-1, 1])
        normalized = np.clip(normalized, -.99, .99)
        normalized = normalized[:, 0]
        rows = [normalized, selected_component]
        if self.null_transformer and self.null_transformer.models_missing_values():
            rows.append(model_missing_values)

        return np.stack(rows, axis=1)  # noqa: PD013

    def _reverse_transform_helper(self, data):
        normalized = np.clip(data[:, 0], -1, 1)
        means = self._bgm_transformer.means_.reshape([-1])
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape([-1])
        selected_component = data[:, 1].round().astype(int)
        selected_component = selected_component.clip(0, self.valid_component_indicator.sum() - 1)
        std_t = stds[self.valid_component_indicator][selected_component]
        mean_t = means[self.valid_component_indicator][selected_component]
        reversed_data = normalized * self.STD_MULTIPLIER * std_t + mean_t

        return reversed_data

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.DataFrame or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        recovered_data = self._reverse_transform_helper(data)
        if self.null_transformer and self.null_transformer.models_missing_values():
            recovered_data = np.stack([recovered_data, data[:, -1]], axis=1)  # noqa: PD013

        return super()._reverse_transform(recovered_data)
