import numpy as np
import pandas as pd
import pytest

import rdt
import tests.datasets

DATA_SIZE = 1000

# Temporary mapping
TRANSFORMER_TO_TYPE = {
    rdt.transformers.null.NullTransformer: None,
    rdt.transformers.boolean.BooleanTransformer: 'boolean',
    rdt.transformers.categorical.CategoricalTransformer: 'categorical',
    rdt.transformers.categorical.OneHotEncodingTransformer: 'categorical',
    rdt.transformers.categorical.LabelEncodingTransformer: 'categorical',
    rdt.transformers.datetime.DatetimeTransformer: 'datetime',
    rdt.transformers.numerical.NumericalTransformer: 'numerical',
    rdt.transformers.numerical.GaussianCopulaTransformer: 'numerical',
}

# Temporary mapping
TRANSFORMER_TO_TRANSFORMED_TYPE = {
    rdt.transformers.null.NullTransformer: None,
    rdt.transformers.boolean.BooleanTransformer: 'float',
    rdt.transformers.categorical.CategoricalTransformer: 'float',
    rdt.transformers.categorical.OneHotEncodingTransformer: 'int',
    rdt.transformers.categorical.LabelEncodingTransformer: 'int',
    rdt.transformers.datetime.DatetimeTransformer: 'float',
    rdt.transformers.numerical.NumericalTransformer: 'float',
    rdt.transformers.numerical.GaussianCopulaTransformer: 'float',
}


def _get_subclasses(obj):
    subclasses = obj.__subclasses__()

    if len(subclasses) == 0:
        return []

    for sc in subclasses:
        subclasses.extend(_get_subclasses(sc))

    return subclasses


def _build_generator_map():
    generators = {}

    for g in tests.datasets.BaseDatasetGenerator.__subclasses__():
        generators[g.DATA_TYPE] = g.__subclasses__()

    return generators


def _validate_input_type(input_type):
    assert input_type is not None


def _find_dataset_generators(data_type, generators):
    if data_type is None or data_type not in generators:
        return []

    return generators[data_type]


def _validate_transformed_data(transformer, transformed_data):
    expected_dtype = TRANSFORMER_TO_TRANSFORMED_TYPE[transformer]
    assert transformed_data.dtype == expected_dtype


def _validate_reverse_transformed_data(transformer, reversed_data, input_dtype):
    expected_dtype = input_dtype
    assert reversed_data.dtype == expected_dtype


def _validate_composition(transformer, reversed_data, input_data):
    # if transformer.is_composition_identity():
    pd.testing.assert_series_equal(reversed_data, input_data, check_dtype=False)


def _test_transformer_with_hypertransformer(transformer, input_data):
    # reverse transformed data using hypertransformer, check that output type is same as input.
    col_name = 'test_col'
    data = pd.DataFrame({col_name: input_data})

    hypertransformer = rdt.HyperTransformer(transformers={
        col_name: {'class': transformer.__name__},
    })
    hypertransformer.fit(data)
    transformed = hypertransformer.transform(data)

    # Expect transformed data to be float.
    for dt in transformed.dtypes:
        assert dt == 'float' or dt == 'int'

    out = hypertransformer.reverse_transform(transformed)
    # Except reverse transformed data to have the same dtype as the original data.
    assert out[col_name].dtype == input_data.dtype


def _test_transformer_with_dataset(transformer, input_data):
    # Fit
    t = transformer()
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

    _validate_composition(transformer, out, input_data)


transformers = _get_subclasses(rdt.transformers.BaseTransformer)
generators = _build_generator_map()


@pytest.mark.parametrize('transformer', transformers)
def test_transformer(subtests, transformer):
    data_type = TRANSFORMER_TO_TYPE[transformer]  # transformer.get_input_type()
    assert data_type is not None

    dataset_generators = _find_dataset_generators(data_type, generators)
    assert len(dataset_generators) > 0

    for dg in dataset_generators:
        with subtests.test(msg="test_transformer_with_dataset", generator=dg):
            input_data = pd.Series(dg.generate(DATA_SIZE))
            _test_transformer_with_dataset(transformer, input_data)
            _test_transformer_with_hypertransformer(transformer, input_data)
