from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers.numerical import BayesGMMTransformer


class TestBayesGMMTransformer(TestCase):

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    def test__fit(self, mock_bgm):
        """Test '_fit'."""
        # Setup
        data = pd.Series(np.random.random(size=100))
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])  # 2 components greater than weight_th

        # Run
        transformer = BayesGMMTransformer(max_clusters=10, weight_threshold=0.005)
        transformer._fit(data)

        # Asserts
        assert transformer._valid_component_indicator.sum() == 2
        assert transformer._number_of_modes == 2
        assert transformer._column_raw_dtype == float

    def test__transform_continuous(self):
        """Test '_transform_continuous'.

        The `transform` method first computes the probability that the
        continuous value came from each component, then samples a component based
        on that probability and finally returns a (value, onehot) tuple, where the
        onehot vector indicates the component that was selected and the value
        is a normalized representation of the continuous value based on the mean
        and std of the selected component.

        This test mocks the gaussian mixture model used to compute the probabilities
        as well as the sampling method; it returns deterministic values to avoid
        randomness in the test. Then, it tests to make sure the probabilities
        are computed correctly and that the normalized value is computed correctly.

        Setup:
            - Create column_transform_info with mocked transformer
            - Mock the BayesianGaussianMixture transformer
               - specify means, covariances, predict_proba
               - means = [0, 10]
               - covariances = [1.0, 11.0]
            - Mock np.random.choice to choose maximum likelihood

        Input:
            - column_transform_info
            - raw_column_data = np.array([0.001, 11.9999, 13.001])

        Output:
            - normalized_value (assert between -1.0, 1.0)
              - assert approx = [0.0, -1.0, 1.0]
            - onehot (assert that it's a one-hot encoding)
              - assert = [[0, 1], [1, 0], [1, 0]]

        Side Effects:
            - assert predict_proba called
            - assert np.random.choice with appropriate probabilities
        """

    def test__transform(self):
        """Test 'transform'."""
        # Setup
        transformer = BayesGMMTransformer(max_clusters=3)
        transformer._bgm_transformer = Mock()

        means = np.array([
            [0.90138867],
            [0.09169366],
            [0.499]
        ])
        transformer._bgm_transformer.means_ = means

        covariances = np.array([
            [[0.09024532]],
            [[0.08587948]],
            [[0.27487667]]
        ])
        transformer._bgm_transformer.covariances_ = covariances

        probabilities = np.array([
            [0.01519528, 0.98480472, 0.],
            [0.01659093, 0.98340907, 0.],
            [0.012744, 0.987256, 0.],
            [0.012744, 0.987256, 0.],
            [0.01391614, 0.98608386, 0.],
            [0.99220664, 0.00779336, 0.],
            [0.99059634, 0.00940366, 0.],
            [0.9941256, 0.0058744, 0.],
            [0.99465502, 0.00534498, 0.],
            [0.99059634, 0.00940366, 0.]
        ])
        transformer._bgm_transformer.predict_proba.return_value = probabilities

        transformer._valid_component_indicator = np.array([True, True, False])
        transformer._max_clusters = 3

        data = pd.Series(np.array([0.01, 0.02, -0.01, -0.01, 0.0, 0.99, 0.97, 1.02, 1.03, 0.97]))

        # Run
        result = transformer._transform(data)

        # Asserts
        expected_continuous = np.array([
            -0.06969212, -0.06116121, -0.08675394, -0.08675394, -0.07822303,
            0.07374234, 0.05709835, 0.09870834, 0.10703034, 0.05709835
        ])
        np.testing.assert_allclose(result['continuous'].to_numpy(), expected_continuous, rtol=0.01)

        expected_discrete = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(result['discrete'].to_numpy(), expected_discrete)

    def test__reverse_transform_helper(self):
        """Test '_inverse_transform_continuous' with sigmas != None.

        The '_inverse_transform_continuous' method should be able to return np.ndarray
        to the appropriate continuous column. However, it currently cannot do so because
        of the way sigmas/st is being passed around. We should look into a less hacky way
        of using this function for TVAE...

        Setup:
            - Mock column_transform_info

        Input:
            - column_data = np.ndarray
              - the first column contains the normalized value
              - the remaining columns correspond to the one-hot
            - sigmas = np.ndarray of floats
            - st = index of the sigmas ndarray

        Output:
            - numpy array containing a single column of continuous values

        Side Effects:
            - None
        """

    def test_reverse_transform(self):
        """Test 'inverse_transform' on a np.ndarray representing one continuous and one
        discrete columns.

        It should use the appropriate '_fit' type for each column and should return
        the corresponding columns. Since we are using the same example as the 'test_transform',
        and these two functions are inverse of each other, the returned value here should
        match the input of that function.

        Setup:
            - Mock _column_transform_info_list
            - Mock _inverse_transform_discrete
            - Mock _inverse_trarnsform_continuous

        Input:
            - column_data = a concatenation of two np.ndarrays
              - the first one refers to the continuous values
                - the first column contains the normalized values
                - the remaining columns correspond to the a one-hot
              - the second one refers to the discrete values
                - the columns correspond to a one-hot
        Output:
            - numpy array containing a discrete column and a continuous column

        Side Effects:
            - _transform_discrete and _transform_continuous should each be called once.
        """
