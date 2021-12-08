from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers.numerical import BayesGMMTransformer, ColumnTransformInfo, SpanInfo


class TestBayesGMMTransformer(TestCase):

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    def test___fit_continuous(self, mock_bgm):
        """Test '_fit_continuous_' on a simple continuous column.

        A 'BayesianGaussianMixture' will be created and fit with the
        'raw_column_data'.

        Setup:
            - Create BayesGMMTransformer with weight_threshold
            - Mock the BayesianGaussianMixture
            - Provide fit method (no-op)
            - Provide weights_ attribute, some above threshold, some below

        Input:
            - column_name = string
            - raw_column_data = numpy array of continuous values

        Output:
            - ColumnTransformInfo
              - Check column name
              - Check that output_dims matches expected (1 + # weights above threshold)

        Side Effects:
            - fit should be called with the data
        """
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])  # 2 non-zero components

        max_clusters = 10
        transformer = BayesGMMTransformer(max_clusters, weight_threshold=0.005)
        info = transformer._fit_continuous('column', np.random.normal((100, 1)))

        assert info.column_name == 'column'
        assert info.transform == bgm_instance
        assert info.output_dimensions == 3
        assert info.output_info[0].dim == 1
        assert info.output_info[0].activation_fn == 'tanh'
        assert info.output_info[1].dim == 2
        assert info.output_info[1].activation_fn == 'softmax'

    def test__fit(self):
        """Test '_fit' on a np.ndarray with one continuous and one discrete columns.

        The '_fit' method should:
            - Set 'self.dataframe' to 'False'
            - Set 'self._column_raw_dtypes' to the appropirate dtypes
            - Use the appropriate '_fit' type for each column'
            - Update 'self.output_info_list', 'self.output_dimensions' and
            'self._column_transform_info_list' appropriately

        Setup:
            - Create BayesGMMTransformer
            - Mock _fit_discrete
            - Mock _fit_continuous

        Input:
            - raw_data = a table with one continuous and one discrete columns.
            - discrete_columns = list with the name of the discrete column

        Output:
            - None

        Side Effects:
            - _fit_discrete and _fit_continuous should each be called once
            - Assigns 'self._column_raw_dtypes' the appropriate dtypes
            - Assigns 'self.output_info_list' the appropriate 'output_info'.
            - Assigns 'self.output_dimensions' the appropriate 'output_dimensions'.
            - Assigns 'self._column_transform_info_list' the appropriate 'column_transform_info'.
        """
        data = pd.DataFrame({
            'x': np.random.random(size=100),
        })

        transformer = BayesGMMTransformer()
        transformer._fit_continuous = Mock()
        transformer._fit_continuous.return_value = ColumnTransformInfo(
            column_name='x', column_type='continuous', transform=None,
            transform_aux=None,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
            output_dimensions=1 + 3
        )

        transformer._fit(data)
        transformer._fit_continuous.assert_called_once()
        assert transformer.output_dimensions == 4

    def test__transform_continuous(self):
        """Test '_transform_continuous'.

        The `transform_continuous` method first computes the probability that the
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
        """Test 'transform' on a dataframe with one continuous and one discrete columns.

        It should use the appropriate '_transform' type for each column and should return
        them concanenated appropriately.

        Setup:
            - Mock _column_transform_info_list
            - Mock _transform_discrete
            - Mock _transform_continuous

        Input:
            - raw_data = a table with one continuous and one discrete columns.

        Output:
            - numpy array containing the transformed two columns

        Side Effects:
            - _transform_discrete and _transform_continuous should each be called once.
        """
        data = pd.DataFrame({
            'x': np.array([0.1, 0.3, 0.5]),
        })

        transformer = BayesGMMTransformer()
        transformer._column_transform_info_list = [
            ColumnTransformInfo(
                column_name='x', column_type='continuous', transform=None,
                transform_aux=None,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
                output_dimensions=1 + 3
            )
        ]

        transformer._transform_continuous = Mock()
        selected_normalized_value = np.array([[0.1], [0.3], [0.5]])
        selected_component_onehot = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        return_value = (selected_normalized_value, selected_component_onehot)
        transformer._transform_continuous.return_value = return_value

        result = transformer._transform(data)
        transformer._transform_continuous.assert_called_once()

        continuous = np.array([0.1, 0.3, 0.5])
        discrete = np.array([0, 0, 1])

        assert result.shape == (3, 2)
        assert (result['continuous'].to_numpy() == continuous).all(), 'continuous'
        assert (result['discrete'].to_numpy() == discrete).all(), 'discrete'

    def test__reverse_transform_continuous(self):
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
