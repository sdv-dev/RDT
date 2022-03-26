from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.performance.datasets import BaseDatasetGenerator
from rdt.transformers import BaseTransformer

DATA_SIZE = 1000
TEST_COL = 'test_col'

PRIMARY_SDTYPES = ['boolean', 'categorical', 'datetime', 'numerical']

# Additional arguments for transformers
TRANSFORMER_ARGS = {
    'BinaryEncoder': {
        'missing_value_replacement': -1,
        'model_missing_values': True
    },
    'UnixTimestampEncoder': {
        'missing_value_replacement': 'mean',
        'model_missing_values': True
    },
    'OptimizedTimestampEncoder': {
        'missing_value_replacement': 'mean',
        'model_missing_values': True
    },
    'FloatFormatter': {
        'missing_value_replacement': 'mean',
        'model_missing_values': True
    },
    'NumericalRoundedBoundedTransformer': {
        'missing_value_replacement': 'mean',
        'model_missing_values': True
    },
    'NumericalBoundedTransformer': {
        'missing_value_replacement': 'mean',
        'model_missing_values': True
    },
    'GaussianNormalizer': {
        'model_missing_values': True
    },
    'ClusterBasedNormalizer': {
        'model_missing_values': True
    },
}

# Mapping of rdt sdtype to dtype
SDTYPE_TO_DTYPES = {
    'boolean': ['b', 'O'],
    'categorical': ['O', 'i', 'f'],
    'datetime': ['M'],
    'numerical': ['f', 'i'],
    'integer': ['i'],
    'float': ['f', 'i'],
    'pii': ['O', 'i', 'f'],
}


def _validate_helper(validator_function, args, steps):
    """Wrap around validation functions to either return a boolean or assert.

    Args:
        validator_function(function):
            The function to validate.
        args (list):
            The args to pass into the function.
        steps (list):
            List of steps that the validation has completed.
    """
    if steps is not None:
        steps.append(validator_function.__name__)

    validator_function(*args)


def _is_valid_transformer(transformer_name):
    """Determine if transformer should be tested or not."""
    return transformer_name != 'IdentityTransformer' and 'Dummy' not in transformer_name


def _get_all_transformers():
    """Get all transformers to be tested."""
    all_transformers = BaseTransformer.get_subclasses()
    return [t for t in all_transformers if _is_valid_transformer(t.__name__)]


def _build_generator_map():
    """Build a map of sdtype to data generator.

    Output:
        dict:
            A mapping of sdtype (str) to a list of data
            generators (rdt.tests.datasets.BaseDatasetGenerator).
    """
    generators = defaultdict(list)

    for generator in BaseDatasetGenerator.get_subclasses():
        generators[generator.SDTYPE].append(generator)

    return generators


def _find_dataset_generators(sdtype, generators):
    """Find the dataset generators for the given sdtype."""
    if sdtype is None:
        primary_generators = []
        for primary_sdtype in PRIMARY_SDTYPES:
            primary_generators.extend(_find_dataset_generators(primary_sdtype, generators))

        return primary_generators

    return generators.get(sdtype, [])


def _validate_dataset_generators(dataset_generators):
    """Check that the number of dataset generators is greater than zero."""
    assert len(dataset_generators) > 0, 'There are no associated dataset generators.'


def _validate_transformed_data(transformer, transformed_data):
    """Check that the transformed data is the expected dtype."""
    expected_sdtypes = transformer.get_output_sdtypes()
    transformed_dtypes = transformed_data.dtypes

    for column, expected_sdtype in expected_sdtypes.items():
        message = f'Column {column} is expected but not found in transformed data.'
        assert column in transformed_data, message
        message = f'Column {column} is not the expected sdtype {expected_sdtype}'
        assert transformed_dtypes[column].kind in SDTYPE_TO_DTYPES[expected_sdtype], message


def _validate_reverse_transformed_data(transformer, reversed_data, input_dtype):
    """Check that the reverse transformed data is the expected dtype.

    Expect that the dtype is equal to the dtype of the input data.
    """
    expected_sdtype = transformer.get_input_sdtype()
    message = f'Reverse transformed data is not the expected sdtype {expected_sdtype}'
    assert reversed_data.dtypes[TEST_COL].kind in SDTYPE_TO_DTYPES[expected_sdtype], message


def _validate_composition(transformer, reversed_data, input_data):
    """Check that the reverse transformed data is equal to the input.

    This is only applicable if the transformer has the composition
    identity property.
    """
    if isinstance(reversed_data, pd.DataFrame):
        reversed_data = reversed_data[TEST_COL]
    elif isinstance(reversed_data, np.ndarray):
        reversed_data = pd.Series(reversed_data)

    if pd.api.types.is_datetime64_any_dtype(reversed_data):
        reversed_data = reversed_data.round('us')

    pd.testing.assert_series_equal(
        reversed_data,
        input_data[TEST_COL],
        check_dtype=False,
        check_exact=False,
        rtol=1e-03,
        obj='Reverse transformed data and input data',
    )


def _test_transformer_with_dataset(transformer_class, input_data, steps):
    """Test the given transformer with the given input data.

    This method verifies the transformed data's dtype, the reverse
    transformed data (if `is_composition_identity`) and the dtype.

    Args:
        transformer_class (rdt.transformers.BaseTransformer):
            The transformer class to test.
        input_data (pandas.Series):
            The data to test on.
        steps (list):
            List of steps that the validation has completed.
    """

    transformer_args = TRANSFORMER_ARGS.get(transformer_class.__name__, {})
    transformer = transformer_class(**transformer_args)
    # Fit
    transformer.fit(input_data, [TEST_COL])

    # Transform
    transformed = transformer.transform(input_data)
    _validate_helper(
        _validate_transformed_data,
        [transformer, transformed],
        steps,
    )

    # Reverse transform
    out = transformer.reverse_transform(transformed)
    _validate_helper(
        _validate_reverse_transformed_data,
        [transformer, out, input_data.dtypes[TEST_COL]],
        steps,
    )

    if transformer.is_composition_identity():
        _validate_helper(
            _validate_composition,
            [transformer, out, input_data],
            steps,
        )


def _validate_hypertransformer_transformed_data(transformed_data):
    """Check that the transformed data is not null and of type float."""
    assert transformed_data.notna().all(axis=None), 'Transformed data has nulls.'

    for dtype in transformed_data.dtypes:
        assert dtype.kind in SDTYPE_TO_DTYPES['numerical'], 'Transformed data is not numerical.'


def _validate_hypertransformer_reverse_transformed_data(transformer, reversed_data):
    """Check that the reverse transformed data has the same dtype as the input."""
    expected_sdtype = transformer().get_input_sdtype()
    message = f'Reversed transformed data is not the expected sdtype {expected_sdtype}'
    assert reversed_data.dtype.kind in SDTYPE_TO_DTYPES[expected_sdtype], message


def _test_transformer_with_hypertransformer(transformer_class, input_data, steps):
    """Test the given transformer in the hypertransformer.

    Run the provided transformer using the hypertransformer using the provided
    input data. Verify that the expected dtypes are returned by transform
    and reverse_transform.

    Args:
        transformer_class (rdt.transformers.BaseTransformer):
            The transformer class to test.
        input_data (pandas.Series):
            The data to test on.
        steps (list):
            List of steps that the validation has completed.
    """
    transformer_args = TRANSFORMER_ARGS.get(transformer_class.__name__, {})
    hypertransformer = HyperTransformer()
    if transformer_args:
        field_transformers = {
            TEST_COL: transformer_class(**transformer_args)
        }

    else:
        field_transformers = {
            TEST_COL: transformer_class
        }

    hypertransformer.detect_initial_config(input_data)
    hypertransformer.update_transformers(field_transformers)
    hypertransformer.fit(input_data)

    transformed = hypertransformer.transform(input_data)
    _validate_helper(
        _validate_hypertransformer_transformed_data,
        [transformed],
        steps
    )

    out = hypertransformer.reverse_transform(transformed)
    _validate_helper(
        _validate_hypertransformer_reverse_transformed_data,
        [transformer_class, out[TEST_COL]],
        steps,
    )


def validate_transformer(transformer, steps=None, subtests=None):
    """Validate that the transformer passes all integration checks.

    Args:
        transformer (rdt.transformer.BaseTransformer):
            The transformer to validate.
        steps (list):
            List of steps that the validation has completed.
        subtests:
            Whether or not to test with subtests.
    """
    input_sdtype = transformer.get_input_sdtype()

    dataset_generators = _find_dataset_generators(input_sdtype, generators)
    _validate_helper(_validate_dataset_generators, [dataset_generators], steps)

    for dg in dataset_generators:
        data = pd.DataFrame({TEST_COL: dg.generate(DATA_SIZE)})

        if subtests:
            with subtests.test(msg=f'test_transformer_with_dataset_{dg}', generator=dg):
                _test_transformer_with_dataset(transformer, data, steps)
                _test_transformer_with_hypertransformer(transformer, data, steps)
        else:
            _test_transformer_with_dataset(transformer, data, steps)
            _test_transformer_with_hypertransformer(transformer, data, steps)


transformers = _get_all_transformers()
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
    validate_transformer(transformer, subtests=subtests)
