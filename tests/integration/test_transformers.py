import numpy as np
import pandas as pd
import pytest

import rdt.transformers
import tests.datasets

DATA_SIZES = [100, 1000, 10000]

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


def get_subclasses(obj):
    subclasses = obj.__subclasses__()

    if len(subclasses) == 0:
        return []

    for sc in subclasses:
        subclasses.extend(get_subclasses(sc))

    return subclasses


def build_generator_map():
    generators = {}

    for g in tests.datasets.BaseDatasetGenerator.__subclasses__():
        generators[g.DATA_TYPE] = g.__subclasses__()

    return generators


def get_transformer_test_cases():
    test_cases = []

    transformers = get_subclasses(rdt.transformers.BaseTransformer)
    generators = build_generator_map()

    for t in transformers:
        for ds in DATA_SIZES:
            dt = TRANSFORMER_TO_TYPE[t]

            if dt is None:
                continue

            gs = generators[dt]

            for g in gs:
                test_cases.append((t, ds, g))

    return test_cases


test_cases = get_transformer_test_cases()


@pytest.mark.parametrize('transformer,data_size,dataset_generator', test_cases)
def test_transformer(transformer, data_size, dataset_generator):
    # Generate input data
    input_data = pd.Series(dataset_generator.generate(data_size))

    # Fit
    t = transformer()
    t.fit(input_data)

    # Transform
    transformed = t.transform(input_data)
    # TODO: Verify transformed data type

    # Reverse transform
    out = t.reverse_transform(transformed)
    # TODO: Verify out data type

    if isinstance(out, pd.DatetimeIndex):
        out = out.to_series(index=pd.RangeIndex(start=0, stop=data_size, step=1))
    if pd.api.types.is_datetime64_any_dtype(out):
        out = out.round('us')
    if isinstance(out, np.ndarray):
        out = pd.Series(out)

    pd.testing.assert_series_equal(out, input_data, check_dtype=False)
