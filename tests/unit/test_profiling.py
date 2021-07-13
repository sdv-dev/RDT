from unittest.mock import Mock

import numpy as np

from rdt.transformers import NumericalTransformer
from tests.performance.datasets import RandomNumericalGenerator
from tests.performance.profiling import profile_transformer


def test_profile_transformer():
    """Test the ``profile_transformer`` function.

    The function should run the ``fit``, ``transform``
    and ``reverse_transform`` method for the provided transformer
    with the dataset created by the provided generator. It should
    also output a DataFrame with the average time and peak memory
    for each method.

    Input:
    - Mock transformer
    - Mock dataset generator
    - transform size of 100

    Side effects:
    - ``fit``, ``transform`` and ``reverse_transform`` should be
    called with correct data

    Output:
    - DataFrame with times and memories
    """
    # Setup
    transformer_mock = Mock(spec_set=NumericalTransformer)
    dataset_gen_mock = Mock(spec_set=RandomNumericalGenerator)
    transformer_mock.transform.return_value = np.zeros(100)
    dataset_gen_mock.generate.return_value = np.ones(100)

    # Run
    profiling_results = profile_transformer(transformer_mock, dataset_gen_mock, 100)

    # Assert
    expected_output_columns = [
        'Fit Time', 'Fit Memory', 'Transform Time', 'Transform Memory',
        'Reverse Transform Time', 'Reverse Transform Memory'
    ]
    assert len(transformer_mock.fit.mock_calls) == 101
    assert len(transformer_mock.transform.mock_calls) == 102
    assert len(transformer_mock.reverse_transform.mock_calls) == 101
    all(np.testing.assert_array_equal(call[1][0], np.ones(100)) for call
        in transformer_mock.fit.mock_calls)
    all(np.testing.assert_array_equal(call[1][0], np.ones(100)) for call
        in transformer_mock.transform.mock_calls)
    all(np.testing.assert_array_equal(call[1][0], np.zeros(100)) for call
        in transformer_mock.reverse_transform.mock_calls)
    expected_output_columns == list(profiling_results.columns)
