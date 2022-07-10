"""Tests for the profiling module."""

from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.performance.datasets import BaseDatasetGenerator
from rdt.performance.profiling import profile_transformer
from rdt.transformers import FloatFormatter


@patch('rdt.performance.profiling.mp')
@patch('rdt.performance.profiling.deepcopy', spec_set=deepcopy)
def test_profile_transformer(deepcopy_mock, multiprocessor_mock):
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
    transformer_mock = Mock(spec_set=FloatFormatter)
    dataset_gen_mock = Mock(spec_set=BaseDatasetGenerator)
    transformer_mock.return_value.transform.return_value = np.zeros(100)
    dataset_gen_mock.generate.return_value = np.ones(100)
    deepcopy_mock.return_value = transformer_mock.return_value

    # Run
    profiling_results = profile_transformer(transformer_mock.return_value,
                                            dataset_gen_mock, 100)

    # Assert
    expected_output_columns = [
        'Fit Time', 'Fit Memory', 'Transform Time', 'Transform Memory',
        'Reverse Transform Time', 'Reverse Transform Memory'
    ]
    assert len(deepcopy_mock.mock_calls) == 10
    assert len(transformer_mock.return_value.fit.mock_calls) == 11
    assert len(transformer_mock.return_value.transform.mock_calls) == 11
    assert len(transformer_mock.return_value.reverse_transform.mock_calls) == 10

    all(np.testing.assert_array_equal(call[1][0], np.ones(100)) for call
        in transformer_mock.fit.mock_calls)
    all(np.testing.assert_array_equal(call[1][0], np.ones(100)) for call
        in transformer_mock.transform.mock_calls)
    all(np.testing.assert_array_equal(call[1][0], np.zeros(100)) for call
        in transformer_mock.reverse_transform.mock_calls)

    assert expected_output_columns == list(profiling_results.index)

    process_mock = multiprocessor_mock.get_context().Process
    fit_call = process_mock.mock_calls[0]
    transform_call = process_mock.mock_calls[3]
    reverse_transform_call = process_mock.mock_calls[6]

    assert fit_call[2]['args'][0] == transformer_mock.return_value.fit
    pd.testing.assert_frame_equal(fit_call[2]['args'][1], pd.DataFrame({'test': np.ones(100)}))
    assert transform_call[2]['args'][0] == transformer_mock.return_value.transform
    pd.testing.assert_frame_equal(
        transform_call[2]['args'][1].reset_index(drop=True),
        pd.DataFrame({'test': np.ones(100)})
    )
    assert reverse_transform_call[2]['args'][0] == transformer_mock.return_value.reverse_transform
    np.testing.assert_array_equal(reverse_transform_call[2]['args'][1], np.zeros(100))
