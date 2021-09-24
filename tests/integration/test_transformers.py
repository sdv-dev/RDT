import numpy as np
import pandas as pd
import pytest

import rdt
import tests.datasets

DATA_SIZE = 1000

# Temporary mapping of transformer to (input_type, output_type)
TRANSFORMER_TO_TYPE = {
    rdt.transformers.null.NullTransformer: (None, None),
    rdt.transformers.boolean.BooleanTransformer: ('boolean', 'numerical'),
    rdt.transformers.categorical.CategoricalTransformer: ('categorical', 'numerical'),
    rdt.transformers.categorical.OneHotEncodingTransformer: ('categorical', 'numerical'),
    rdt.transformers.categorical.LabelEncodingTransformer: ('categorical', 'numerical'),
    rdt.transformers.datetime.DatetimeTransformer: ('datetime', 'numerical'),
    rdt.transformers.numerical.NumericalTransformer: ('numerical', 'numerical'),
    rdt.transformers.numerical.GaussianCopulaTransformer: ('numerical', 'numerical'),
}

# Temporary mapping of rdt data type to dtype
DATA_TYPE_TO_DTYPES = {
    'boolean': ['b', 'O'],
    'categorical': ['O', 'i', 'f'],
    'datetime': ['M'],
    'numerical': ['f', 'i'],
}


# Temporary method, will be removed after hypertransformer is updated.
def _get_subclasses(obj):
    """Get all subclasses of the given class."""
    subclasses = obj.__subclasses__()

    if len(subclasses) == 0:
        return []

    for sc in subclasses:
        subclasses.extend(_get_subclasses(sc))

    return subclasses


def _build_generator_map():
    """Build a map of data type to data generator.

    Output:
        dict:
            A mapping of data type (str) to a list of data
            generators (rdt.tests.datasets.BaseDatasetGenerator).
    """
    generators = {}

    for g in tests.datasets.BaseDatasetGenerator.__subclasses__():
        generators[g.DATA_TYPE] = g.__subclasses__()

    return generators


def _find_dataset_generators(data_type, generators):
    """Find the dataset generators for the given data_type."""
    if data_type is None or data_type not in generators:
        return []

    return generators[data_type]


def _validate_input_type(input_type):
    """Check that the transformer input type is not null."""
    assert input_type is not None


def _validate_transformed_data(transformer, transformed_data):
    """Check that the transformed data is the expected dtype."""
    # TODO: Update to use `get_output_types`
    expected_data_type = TRANSFORMER_TO_TYPE[transformer][1]
    assert transformed_data.dtype.kind in DATA_TYPE_TO_DTYPES[expected_data_type]


def _validate_reverse_transformed_data(transformer, reversed_data, input_dtype):
    """Check that the reverse transformed data is the expected dtype.

    Expect that the dtype is equal to the dtype of the input data.
    """
    expected_data_type = TRANSFORMER_TO_TYPE[transformer][0]
    assert reversed_data.dtype.kind in DATA_TYPE_TO_DTYPES[expected_data_type]


def _validate_composition(reversed_data, input_data):
    """Check that the reverse transformed data is equal to the input.

    This is only applicable if the transformer has the composition
    identity property.
    """
    # TODO: only do if transformer.is_composition_identity():
    pd.testing.assert_series_equal(reversed_data, input_data, check_dtype=False)


def _test_transformer_with_dataset(transformer, input_data):
    """Test the given transformer with the given input data.

    This method verifies the transformed data's dtype, the reverse
    transformed data (if `is_composition_identity`) and the dtype.

    Args:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to test.
        input_data (pandas.Series):
            The data to test on.
    """
    t = transformer()
    # Fit
    t.fit(input_data)

    # Transform
    transformed = t.transform(input_data)
    _validate_transformed_data(transformer, transformed)

    # Reverse transform
    out = t.reverse_transform(transformed)
    _validate_reverse_transformed_data(transformer, out, input_data.dtype)

    if isinstance(out, pd.DatetimeIndex):
        out = out.to_series(index=pd.RangeIndex(start=0, stop=DATA_SIZE, step=1))
    if pd.api.types.is_datetime64_any_dtype(out):
        out = out.round('us')
    if isinstance(out, np.ndarray):
        out = pd.Series(out)

    _validate_composition(out, input_data)


def _validate_hypertransformer_transformed_data(transformed_data):
    """Check that the transformed data is of type float."""
    for dt in transformed_data.dtypes:
        assert dt.kind in DATA_TYPE_TO_DTYPES['numerical']


def _validate_hypertransformer_reversed_transformed_data(transformer, reversed_data):
    """Check that the reverse transformed data has the same dtype as the input."""
    expected_data_type = TRANSFORMER_TO_TYPE[transformer][0]
    assert reversed_data.dtype.kind in DATA_TYPE_TO_DTYPES[expected_data_type]


def _test_transformer_with_hypertransformer(transformer, input_data):
    """Test the given transformer in the hypertransformer.

    Run the provided transformer using the hypertransformer using the provided
    input data. Verify that the expected dtypes are returned by transform
    and reverse_transform.

    Args:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to test.
        input_data (pandas.Series):
            The data to test on.
    """
    # reverse transformed data using hypertransformer, check that output type is same as input.
    col_name = 'test_col'
    data = pd.DataFrame({col_name: input_data})

    hypertransformer = rdt.HyperTransformer(transformers={
        col_name: {'class': transformer.__name__},
    })
    hypertransformer.fit(data)

    transformed = hypertransformer.transform(data)
    _validate_hypertransformer_transformed_data(transformed)

    out = hypertransformer.reverse_transform(transformed)
    _validate_hypertransformer_reversed_transformed_data(transformer, out[col_name])


transformers = _get_subclasses(rdt.transformers.BaseTransformer)
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
    input_data_type = TRANSFORMER_TO_TYPE[transformer][0]  # transformer.get_input_type()
    assert input_data_type is not None

    dataset_generators = _find_dataset_generators(input_data_type, generators)
    assert len(dataset_generators) > 0

    for dg in dataset_generators:
        with subtests.test(msg="test_transformer_with_dataset", generator=dg):
            input_data = pd.Series(dg.generate(DATA_SIZE))
            _test_transformer_with_dataset(transformer, input_data)
            _test_transformer_with_hypertransformer(transformer, input_data)
