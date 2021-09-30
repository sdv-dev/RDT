import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers import BaseTransformer
from tests.datasets import BaseDatasetGenerator

DATA_SIZE = 1000
TEST_COL = 'test_col'

# Mapping of rdt data type to dtype
DATA_TYPE_TO_DTYPES = {
    'boolean': ['b', 'O'],
    'categorical': ['O', 'i', 'f'],
    'datetime': ['M'],
    'numerical': ['f', 'i'],
    'integer': ['i'],
    'float': ['f', 'i'],
}


def _build_generator_map():
    """Build a map of data type to data generator.

    Output:
        dict:
            A mapping of data type (str) to a list of data
            generators (rdt.tests.datasets.BaseDatasetGenerator).
    """
    generators = {}

    for g in BaseDatasetGenerator.__subclasses__():
        generators[g.DATA_TYPE] = g.__subclasses__()

    return generators


def _find_dataset_generators(data_type, generators):
    """Find the dataset generators for the given data_type."""
    return generators.get(data_type, [])


def _validate_input_type(input_type):
    """Check that the transformer input type is not null."""
    assert input_type is not None


def _validate_transformed_data(transformer, transformed_data):
    """Check that the transformed data is the expected dtype."""
    expected_data_types = transformer.get_output_types()
    transformed_dtypes = transformed_data.dtypes

    for column, expected_data_type in expected_data_types.items():
        assert column in transformed_data
        assert transformed_dtypes[column].kind in DATA_TYPE_TO_DTYPES[expected_data_type]


def _validate_reverse_transformed_data(transformer, reversed_data, input_dtype):
    """Check that the reverse transformed data is the expected dtype.

    Expect that the dtype is equal to the dtype of the input data.
    """
    expected_data_type = transformer.get_input_type()
    assert reversed_data.dtypes[TEST_COL].kind in DATA_TYPE_TO_DTYPES[expected_data_type]


def _validate_composition(transformer, reversed_data, input_data):
    """Check that the reverse transformed data is equal to the input.

    This is only applicable if the transformer has the composition
    identity property.
    """
    if not transformer.is_composition_identity():
        return

    if isinstance(reversed_data, pd.DataFrame):
        reversed_data = reversed_data[TEST_COL]
    if isinstance(reversed_data, pd.DatetimeIndex):
        reversed_data = reversed_data.to_series(
            index=pd.RangeIndex(start=0, stop=DATA_SIZE, step=1)
        )
    if isinstance(reversed_data, np.ndarray):
        reversed_data = pd.Series(reversed_data)
    if pd.api.types.is_datetime64_any_dtype(reversed_data):
        reversed_data = reversed_data.round('us')

    pd.testing.assert_series_equal(
        reversed_data,
        input_data[TEST_COL],
        check_dtype=False,
        check_exact=False,
        rtol=1e-03,
    )


def _test_transformer_with_dataset(transformer_class, input_data):
    """Test the given transformer with the given input data.

    This method verifies the transformed data's dtype, the reverse
    transformed data (if `is_composition_identity`) and the dtype.

    Args:
        transformer_class (rdt.transformers.BaseTransformer):
            The transformer class to test.
        input_data (pandas.Series):
            The data to test on.
    """
    transformer = transformer_class()
    # Fit
    transformer.fit(input_data, [TEST_COL])

    # Transform
    transformed = transformer.transform(input_data)
    _validate_transformed_data(transformer, transformed)

    # Reverse transform
    out = transformer.reverse_transform(transformed)
    _validate_reverse_transformed_data(transformer, out, input_data.dtypes[TEST_COL])

    _validate_composition(transformer, out, input_data)


def _validate_hypertransformer_transformed_data(transformed_data):
    """Check that the transformed data is of type float."""
    for dt in transformed_data.dtypes:
        assert dt.kind in DATA_TYPE_TO_DTYPES['numerical']


def _validate_hypertransformer_reversed_transformed_data(transformer, reversed_data):
    """Check that the reverse transformed data has the same dtype as the input."""
    expected_data_type = transformer().get_input_type()
    assert reversed_data.dtype.kind in DATA_TYPE_TO_DTYPES[expected_data_type]


def _test_transformer_with_hypertransformer(transformer_class, input_data):
    """Test the given transformer in the hypertransformer.

    Run the provided transformer using the hypertransformer using the provided
    input data. Verify that the expected dtypes are returned by transform
    and reverse_transform.

    Args:
        transformer_class (rdt.transformers.BaseTransformer):
            The transformer class to test.
        input_data (pandas.Series):
            The data to test on.
    """
    hypertransformer = HyperTransformer(transformers={
        TEST_COL: {'class': transformer_class.__name__},
    })
    hypertransformer.fit(input_data)

    transformed = hypertransformer.transform(input_data)
    _validate_hypertransformer_transformed_data(transformed)

    out = hypertransformer.reverse_transform(transformed)
    _validate_hypertransformer_reversed_transformed_data(transformer_class, out[TEST_COL])


transformers = BaseTransformer.get_subclasses()
generators = _build_generator_map()


@pytest.mark.parametrize('transformer', transformers)
def test_transformer(subtests, transformer):
    """Test the transformer end-to-end.

    Test the transformer end-to-end with at least one generated dataset. Test
    both the transformer by itself, and by running in the hypertransformer.

    Args:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to test.
    """
    input_data_type = transformer.get_input_type()
    assert input_data_type is not None

    dataset_generators = _find_dataset_generators(input_data_type, generators)
    assert len(dataset_generators) > 0

    for dg in dataset_generators:
        with subtests.test(msg="test_transformer_with_dataset", generator=dg):
            input_data = pd.Series(dg.generate(DATA_SIZE))

            data = pd.DataFrame({TEST_COL: input_data})
            _test_transformer_with_dataset(transformer, data)
            _test_transformer_with_hypertransformer(transformer, data)
