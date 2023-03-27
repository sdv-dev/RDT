"""Integration tests for the HyperTransformer."""

import re

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer, get_demo
from rdt.errors import ConfigNotSetError, InvalidConfigError, InvalidDataError, NotFittedError
from rdt.transformers import (
    AnonymizedFaker, BaseTransformer, BinaryEncoder, ClusterBasedNormalizer, FloatFormatter,
    FrequencyEncoder, LabelEncoder, OneHotEncoder, RegexGenerator, UnixTimestampEncoder,
    get_default_transformer, get_default_transformers)
from rdt.transformers.pii.anonymizer import PseudoAnonymizedFaker


class DummyTransformerNumerical(BaseTransformer):

    INPUT_SDTYPE = 'categorical'

    def _fit(self, data):
        pass

    def _transform(self, data):
        return data.astype(float)

    def _reverse_transform(self, data):
        return data.astype(str)


class DummyTransformerNotMLReady(BaseTransformer):

    INPUT_SDTYPE = 'datetime'

    def __init__(self):
        super().__init__()
        self.output_properties = {
            None: {'sdtype': 'datetime', 'next_transformer': FrequencyEncoder()}
        }

    def _fit(self, data):
        pass

    def _transform(self, data):
        # Stringify input data
        return data.astype(str)

    def _reverse_transform(self, data):
        return data.astype('datetime64')


TEST_DATA_INDEX = [4, 6, 3, 8, 'a', 1.0, 2.0, 3.0]


def get_input_data():
    datetimes = pd.to_datetime([
        '2010-02-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
        '2010-01-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
    ])
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'categorical': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a'],
        'bool': [False, False, False, True, False, False, True, False],
        'datetime': datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)

    return data


def get_transformed_data():
    datetimes = [
        1.264982e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18,
        1.262304e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18
    ]
    return pd.DataFrame({
        'integer': [1., 2., 1., 3., 1., 4., 2., 3.],
        'float': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'categorical': [0.3125, 0.3125, .8125, 0.8125, 0.3125, 0.8125, 0.3125, 0.3125],
        'bool': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'datetime': datetimes,
        'names': [0.3125, 0.75, 0.75, 0.3125, 0.3125, 0.9375, 0.3125, 0.3125]
    }, index=TEST_DATA_INDEX)


def get_reversed_data():
    data = get_input_data()
    data['bool'] = data['bool'].astype('object')
    return data


def test_hypertransformer_default_inputs():
    """Test the HyperTransformer with default parameters.

    This tests that if default parameters are provided to the HyperTransformer,
    the ``default_transformers`` method will be used to determine which
    transformers to use for each field.

    Input:
        - A dataframe with every sdtype.
        - A fixed random seed to guarantee the samle values are null.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    datetimes = pd.to_datetime([
        np.nan,
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
        '2010-01-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
    ])
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1, 0.4, np.nan, 0.3],
        'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
        'bool': [False, np.nan, False, True, False, np.nan, True, False],
        'datetime': datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_datetimes = [
        1.263069e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18,
        1.262304e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18
    ]
    expected_transformed = pd.DataFrame({
        'integer': [1., 2., 1., 3., 1., 4., 2., 3.],
        'float': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'categorical': [0.3125, 0.3125, 0.9375, 0.75, 0.3125, 0.75, 0.3125, 0.3125],
        'bool': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'datetime': expected_datetimes,
        'names': [0.3125, 0.75, 0.75, 0.3125, 0.3125, 0.9375, 0.3125, 0.3125]
    }, index=TEST_DATA_INDEX)
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    reversed_datetimes = pd.to_datetime([
        '2010-01-09 20:34:17.142857216',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
        '2010-01-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
    ])
    expected_reversed = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, 0.20000000000000004, 0.1, 0.4, 0.20000000000000004, 0.3],
        'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
        'bool': [False, False, False, True, False, False, True, False],
        'datetime': reversed_datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)
    for row in range(reverse_transformed.shape[0]):
        for column in range(reverse_transformed.shape[1]):
            expected = expected_reversed.iloc[row, column]
            actual = reverse_transformed.iloc[row, column]
            assert pd.isna(actual) or expected == actual

    assert isinstance(ht.field_transformers['integer'], FloatFormatter)
    assert isinstance(ht.field_transformers['float'], FloatFormatter)
    assert isinstance(ht.field_transformers['categorical'], FrequencyEncoder)
    assert isinstance(ht.field_transformers['bool'], BinaryEncoder)
    assert isinstance(ht.field_transformers['datetime'], UnixTimestampEncoder)
    assert isinstance(ht.field_transformers['names'], FrequencyEncoder)

    get_default_transformers.cache_clear()
    get_default_transformer.cache_clear()


def test_hypertransformer_field_transformers():
    """Test the HyperTransformer with ``field_transformers`` provided.

    This tests that the transformers specified in the ``field_transformers``
    argument are used. Any output of a transformer that is not ML ready (not
    in the ``_transform_output_sdtypes`` list) should be recursively transformed
    till it is.

    Setup:
        - The datetime column is set to use a dummy transformer that stringifies
        the input. That output is then set to use the categorical transformer.

    Input:
        - A dict mapping each field to a transformer.
        - A dataframe with every sdtype.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    config = {
        'sdtypes': {
            'integer': 'numerical',
            'float': 'numerical',
            'categorical': 'categorical',
            'bool': 'boolean',
            'datetime': 'datetime',
            'names': 'categorical'
        },
        'transformers': {
            'integer': FloatFormatter(missing_value_replacement='mean'),
            'float': FloatFormatter(missing_value_replacement='mean'),
            'categorical': FrequencyEncoder(),
            'bool': BinaryEncoder(missing_value_replacement='mode'),
            'datetime': DummyTransformerNotMLReady(),
            'names': FrequencyEncoder()
        }
    }

    data = get_input_data()

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.set_config(config)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_transformed = get_transformed_data()
    transformed_datetimes = [0.8125, 0.8125, 0.3125, 0.3125, 0.3125, 0.8125, 0.3125, 0.3125]
    expected_transformed['datetime'] = transformed_datetimes
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    expected_reversed = get_reversed_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)


def test_single_category():
    """Test that categorical variables with a single value are supported."""
    # Setup
    ht = HyperTransformer()
    data = pd.DataFrame({
        'a': ['a', 'a', 'a']
    })

    # Run
    ht.detect_initial_config(data)
    ht.update_transformers(column_name_to_transformer={
        'a': OneHotEncoder()
    })
    ht.fit(data)
    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(data, reverse)


def test_categorical_encoders_with_booleans():
    """Test that categorical encoders support boolean values."""
    # Setup
    config = {
        'sdtypes': {
            'email_confirmed': 'boolean',
            'subscribed': 'boolean',
            'paid': 'boolean',
        },
        'transformers': {
            'email_confirmed': FrequencyEncoder(),
            'subscribed': OneHotEncoder(),
            'paid': LabelEncoder(),
        }
    }

    ht = HyperTransformer()

    # Run and Assert
    ht.set_config(config)


def test_dtype_category():
    """Test that categorical variables of dtype category are supported."""
    # Setup
    data = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(reverse, data)


def test_multiple_fits():
    """HyperTransformer should be able to be used multiple times.

    Fitting, transforming and reverse transforming should produce the same results when
    called on the same data multiple times.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed1 = ht.transform(data)
    reversed1 = ht.reverse_transform(transformed1)

    ht.detect_initial_config(data)
    ht.fit(data)
    transformed2 = ht.transform(data)
    reversed2 = ht.reverse_transform(transformed2)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reversed1, reversed2)


def test_multiple_fits_different_data():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1.0, 0.0, 0.0]})
    new_data = pd.DataFrame({'col2': [1, 2, 3], 'col1': [1.0, 0.0, 0.0]})
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    ht.detect_initial_config(new_data)
    ht.fit(new_data)
    transformed1 = ht.transform(new_data)
    transformed2 = ht.transform(new_data)
    reverse1 = ht.reverse_transform(transformed1)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    expected_transformed = pd.DataFrame(
        {'col2': [1., 2., 3.], 'col1': [1.0, 0.0, 0.0]})
    pd.testing.assert_frame_equal(transformed1, expected_transformed)
    pd.testing.assert_frame_equal(transformed2, expected_transformed)
    pd.testing.assert_frame_equal(reverse1, new_data)
    pd.testing.assert_frame_equal(reverse2, new_data)


def test_multiple_fits_different_columns():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1.0, 0.0, 0.0]})
    new_data = pd.DataFrame({'col3': [1, 2, 3], 'col4': [1.0, 0.0, 0.0]})
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    ht.detect_initial_config(new_data)
    ht.fit(new_data)
    transformed1 = ht.transform(new_data)
    transformed2 = ht.transform(new_data)
    reverse1 = ht.reverse_transform(transformed1)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    expected_transformed = pd.DataFrame(
        {'col3': [1., 2., 3.], 'col4': [1.0, 0.0, 0.0]})
    pd.testing.assert_frame_equal(transformed1, expected_transformed)
    pd.testing.assert_frame_equal(transformed2, expected_transformed)
    pd.testing.assert_frame_equal(reverse1, new_data)
    pd.testing.assert_frame_equal(reverse2, new_data)


def test_multiple_fits_with_set_config():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.set_config(config={
        'sdtypes': {'integer': 'categorical'},
        'transformers': {'integer': FrequencyEncoder()}
    })
    ht.fit(data)
    transformed1 = ht.transform(data)
    reverse1 = ht.reverse_transform(transformed1)

    ht.fit(data)
    transformed2 = ht.transform(data)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reverse1, reverse2)


def test_multiple_detect_configs_with_set_config():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed1 = ht.transform(data)
    reverse1 = ht.reverse_transform(transformed1)

    ht.set_config(config={
        'sdtypes': {'integers': 'categorical'},
        'transformers': {'integers': FrequencyEncoder()}
    })

    ht.detect_initial_config(data)
    ht.fit(data)
    transformed2 = ht.transform(data)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reverse1, reverse2)


def test_detect_initial_config_doesnt_affect_fit():
    """HyperTransformer should fit the same way regardless of ``detect_initial_config``.

    Calling the ``detect_initial_config`` method should not affect the results of ``fit``,
    ``transform`` or ``reverse_transform``.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed1 = ht.transform(data)
    reversed1 = ht.reverse_transform(transformed1)

    ht.detect_initial_config(data)
    ht.fit(data)
    transformed2 = ht.transform(data)
    reversed2 = ht.reverse_transform(transformed1)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reversed1, reversed2)


def test_multiple_detects():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = pd.DataFrame({'col2': [1, 2, 3], 'col1': [1.0, 0.0, 0.0]})
    new_data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.detect_initial_config(new_data)
    ht.fit(new_data)
    transformed = ht.transform(new_data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(transformed, get_transformed_data())
    pd.testing.assert_frame_equal(reverse, get_reversed_data())


def test_transform_without_fit():
    """HyperTransformer should raise an error when transforming without fitting."""
    # Setup
    data = pd.DataFrame({'column': [1, 2, 3]})
    ht = HyperTransformer()
    ht.detect_initial_config(data)

    # Run / Assert
    with pytest.raises(NotFittedError):
        ht.transform(data)


def test_fit_data_different_than_detect():
    """HyperTransformer should raise an error when transforming without fitting."""
    # Setup
    ht = HyperTransformer()
    detect_data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    data = pd.DataFrame({'col1': [1, 2], 'col3': ['a', 'b']})

    # Run / Assert
    error_msg = re.escape(
        'The data you are trying to fit has different columns than the original '
        "detected data (unknown columns: ['col3']). Column names and their "
        "sdtypes must be the same. Use the method 'get_config()' to see the expected "
        'values.'
    )
    ht.detect_initial_config(detect_data)
    with pytest.raises(InvalidDataError, match=error_msg):
        ht.fit(data)


def test_transform_without_fitting():
    """HyperTransformer shouldn't transform when fit hasn't been called yet."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    error_msg = (
        'The HyperTransformer is not ready to use. Please fit your data first using '
        "'fit' or 'fit_transform'."
    )
    with pytest.raises(NotFittedError, match=error_msg):
        ht.transform(data)


def test_transform_without_refitting():
    """HyperTransformer shouldn't transform when a new config hasn't been fitted."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    ht.fit(data)
    ht.update_sdtypes({'col1': 'categorical'})
    error_msg = (
        'The HyperTransformer is not ready to use. Please fit your data first using '
        "'fit' or 'fit_transform'."
    )
    with pytest.raises(NotFittedError, match=error_msg):
        ht.transform(data)


def test_transform_without_config():
    """HyperTransformer shouldn't transform when a config hasn't been set."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    ht = HyperTransformer()

    # Run / Assert
    error_msg = (
        "No config detected. Set the config using 'set_config' or pre-populate "
        "it automatically from your data using 'detect_initial_config' prior to "
        'fitting your data.'
    )
    with pytest.raises(ConfigNotSetError, match=error_msg):
        ht.transform(data)


def test_transform_unseen_columns():
    """HyperTransformer shouldn't transform when the data wasn't seen during fit."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    different_data = pd.DataFrame({'col3': [1, 2]})
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    ht.fit(data)
    error_msg = error_msg = (
        'The data you are trying to transform has different columns than the original data. '
        'Column names and their sdtypes must be the same.'
    )
    with pytest.raises(InvalidDataError, match=error_msg):
        ht.transform(different_data)


def test_update_sdtypes_incorrect_columns():
    """HyperTransformer should crash when update_sdytpes is passed non-existing columns."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    column_name_to_sdtype = {'col3': [1, 2]}
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    error_msg = error_msg = re.escape(
        "Invalid column names: ['col3']. These columns do not exist in the "
        "config. Use 'set_config()' to write and set your entire config at once."
    )
    with pytest.raises(InvalidConfigError, match=error_msg):
        ht.update_sdtypes(column_name_to_sdtype)


def test_update_sdtypes_incorrect_sdtype():
    """HyperTransformer should crash when update_sdytpes is passed non-existing columns."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    column_name_to_sdtype = {'col1': 'unexpected'}
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    error_msg = error_msg = re.escape(
        "Invalid sdtypes: ['unexpected']. If you are trying to use a "
        'premium sdtype, contact info@sdv.dev about RDT Add-Ons.'
    )
    with pytest.raises(InvalidConfigError, match=error_msg):
        ht.update_sdtypes(column_name_to_sdtype)


def test_transform_subset():
    """Test the ``transform_subset`` method.

    The method should return a ``pandas.DataFrame`` with the subset of columns transformed.

    Setup:
        - Detect the config and fit the data.

    Input:
        - A ``pandas.DataFrame`` with a subset of the fitted columns.

    Ouput:
        - A ``pandas.DataFrame`` with the subset transformed.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    subset = data[['col1']]
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)

    # Run
    transformed = ht.transform_subset(subset)

    # Assert
    expected = pd.DataFrame({'col1': [1., 2.]})
    pd.testing.assert_frame_equal(transformed, expected)


def test_reverse_transform_subset():
    """Test the ``reverse_transform_subset`` method.

    The method should return a ``pandas.DataFrame`` with the subset of columns reverse transformed.

    Setup:
        - Detect the config and fit the data.

    Input:
        - A ``pandas.DataFrame`` with a subset of the output columns.

    Ouput:
        - A ``pandas.DataFrame`` with the subset reverse transformed.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    subset = pd.DataFrame({'col1': [1, 2]})
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)

    # Run
    reverse_transformed = ht.reverse_transform_subset(subset)

    # Assert
    expected = pd.DataFrame({'col1': [1, 2]})
    pd.testing.assert_frame_equal(reverse_transformed, expected)


def test_hyper_transformer_with_supported_sdtypes():
    """Test the ``HyperTransformer`` that supports multiple ``sdtypes`` for a ``Transformer``.

    Test that the ``HyperTransformer`` works with ``get_supported_sdtypes`` allowing us
    to asign different transformer to a ``sdtype``. For example, a ``FrequencyEncoder`` to
    a ``boolean`` sdtype.

    Setup:
        - Dataframe with multiple datatypes.
        - Instance of ``HyperTransformer``.
        - Update the transformer for ``boolean`` sdtype to ``FrequencyEncoder()``.

    Run:
        - Run end to end the ``hypertransformer``.

    Assert:
        - Assert that the ``FerquencyEncoder`` is used for the ``boolean`` data.
    """
    # Setup
    data = pd.DataFrame({
        'user': ['John', 'Doe', 'John Doe', 'Doe John'],
        'id': list(range(4)),
        'subscribed': [True, False, True, False]
    })

    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.update_transformers_by_sdtype(
        sdtype='boolean',
        transformer=FrequencyEncoder(add_noise=True)
    )

    # Run
    transformed = ht.fit_transform(data)
    ht.reverse_transform(transformed)

    for transformer in ht.get_config()['transformers'].values():
        assert not isinstance(transformer, BinaryEncoder)


def test_hyper_transformer_reverse_transform_subset_and_generators():
    """Test the ``HyperTransformer`` with ``reverse_transform_subset``.

    Test that when calling ``reverse_transform_subset`` and there are ``generators`` like
    ``AnonymizedFaker`` or ``RegexGenerator`` those are not being used in the ``subset``, and
    also any other transformer which can't transform the given columns.

    Setup:
        - DataFrame with multiple datatypes.
        - Instance of HyperTransformer.
        - Add ``pii`` using ``AnonymizedFaker`` and ``RegexGenerator``.

    Run:
        - Use ``fit_transform`` then ``revese_transform_subsample``.

    Assert:
        - Assert that the ``reverse_transformed`` data does not contain any additional columns
          but the expected one.
    """
    # Setup
    customers = get_demo()
    customers['id'] = ['ID_a', 'ID_b', 'ID_c', 'ID_d', 'ID_e']

    # Create a config
    ht = HyperTransformer()
    ht.detect_initial_config(customers)

    # credit_card and id are pii and text columns
    ht.update_sdtypes({
        'credit_card': 'pii',
        'id': 'text'
    })

    ht.update_transformers({
        'credit_card': AnonymizedFaker(),
        'id': RegexGenerator(regex_format='id_[a-z]')
    })

    # Run
    ht.fit(customers)
    transformed = ht.transform(customers)
    reverse_transformed = ht.reverse_transform_subset(transformed[['last_login']])

    # Assert
    expected_transformed_columns = [
        'last_login',
        'email_optin',
        'age',
        'dollars_spent'
    ]
    assert all(expected_transformed_columns == transformed.columns)
    assert reverse_transformed.columns == ['last_login']


def test_set_config_with_supported_sdtypes():
    """Test when setting config with an sdtype that is supported by the transformer."""
    # Setup
    config = {
        'transformers': {
            'boolean_col': FrequencyEncoder(add_noise=True),
        },
        'sdtypes': {
            'boolean_col': 'boolean'
        }
    }
    ht = HyperTransformer()

    # Run and Assert
    ht.set_config(config)


def test_hyper_transformer_chained_transformers():
    """Test when a transformer is chained to another transformer.

    When the specified transformer indicates a next transformer, they should each be applied in
    order during the transform step, and then reversed during the reverse_transform.
    """
    # Setup
    class DoublingTransformer(BaseTransformer):
        INPUT_SDTYPE = 'numerical'

        def _fit(self, data):
            ...

        def _transform(self, data):
            return data * 2

        def _reverse_transform(self, data):
            return data / 2

    transformer3 = DoublingTransformer()
    transformer2 = DoublingTransformer()
    transformer2.output_properties[None]['next_transformer'] = transformer3
    transformer1 = DoublingTransformer()
    transformer1.output_properties[None]['next_transformer'] = transformer2

    ht = HyperTransformer()
    data = pd.DataFrame({'col': [1., 2, -1, 3, 1]})

    # Run and Assert
    ht.set_config({
        'sdtypes': {'col': 'numerical'},
        'transformers': {'col': transformer1}
    })
    ht.fit(data)

    transformed = ht.transform(data)
    expected_transform = pd.DataFrame({'col': [8., 16, -8, 24, 8]})
    pd.testing.assert_frame_equal(transformed, expected_transform)

    reverse_transformed = ht.reverse_transform(transformed)
    pd.testing.assert_frame_equal(reverse_transformed, data)


def test_hyper_transformer_chained_transformers_various_transformers():
    """Test when a transformer is chained to another transformer.

    When the specified transformer indicates a next transformer, they should each be applied in
    order during the transform step, and then reversed during the reverse_transform.
    """
    # Setup
    class AB(BaseTransformer):
        INPUT_SDTYPE = 'categorical'

        def _fit(self, data):
            self.output_properties = {
                None: {'sdtype': 'categorical', 'next_transformer': None},
                'a': {'sdtype': 'categorical', 'next_transformer': CD()},
                'b': {'sdtype': 'categorical', 'next_transformer': None},
            }

        def _transform(self, data):
            new_data = pd.DataFrame({f'{self.column_prefix}': data})
            new_data[f'{self.column_prefix}.a'] = data + 'a'
            new_data[f'{self.column_prefix}.b'] = data + 'b'
            return new_data

        def _reverse_transform(self, data):
            new_data = pd.DataFrame()
            new_data[f'{self.column_prefix}'] = data[f'{self.column_prefix}.a'].str[:-1]
            return new_data

    class CD(BaseTransformer):
        INPUT_SDTYPE = 'categorical'

        def _fit(self, data):
            self.output_properties = {
                'c': {'sdtype': 'categorical', 'next_transformer': None},
                'd': {'sdtype': 'categorical', 'next_transformer': E()}
            }

        def _transform(self, data):
            new_data = pd.DataFrame()
            new_data[f'{self.column_prefix}.c'] = data + 'c'
            new_data[f'{self.column_prefix}.d'] = data + 'd'
            return new_data

        def _reverse_transform(self, data):
            new_data = pd.DataFrame()
            new_data[f'{self.column_prefix}'] = data[f'{self.column_prefix}.c'].str[:-1]
            return new_data

    class E(BaseTransformer):
        INPUT_SDTYPE = 'categorical'

        def _fit(self, data):
            self.output_properties = {
                None: {'sdtype': 'categorical', 'next_transformer': None},
                'e': {'sdtype': 'categorical', 'next_transformer': None}
            }

        def _transform(self, data):
            new_data = pd.DataFrame({f'{self.column_prefix}': data})
            new_data[f'{self.column_prefix}.e'] = data + 'e'
            return new_data

        def _reverse_transform(self, data):
            new_data = pd.DataFrame()
            new_data[f'{self.column_prefix}'] = data[f'{self.column_prefix}.e'].str[:-1]
            return new_data

    ht = HyperTransformer()
    data = pd.DataFrame({
        'col': ['a', 'b', 'c'],
        'col.a': ['1', '2', '3'],
        'col#': ['_', '_', '_']
    })

    # Run and Assert
    ht.set_config({
        'sdtypes': {'col': 'categorical', 'col.a': 'categorical', 'col#': 'categorical'},
        'transformers': {'col': AB(), 'col.a': AB(), 'col#': E()}
    })
    ht.fit(data)
    transformed = ht.transform(data)
    expected = pd.DataFrame({
        'col##': ['a', 'b', 'c'],
        'col##.a.c': ['aac', 'bac', 'cac'],
        'col##.a.d': ['aad', 'bad', 'cad'],
        'col##.a.d.e': ['aade', 'bade', 'cade'],
        'col##.b': ['ab', 'bb', 'cb'],
        'col.a': ['1', '2', '3'],
        'col.a.a.c': ['1ac', '2ac', '3ac'],
        'col.a.a.d': ['1ad', '2ad', '3ad'],
        'col.a.a.d.e': ['1ade', '2ade', '3ade'],
        'col.a.b': ['1b', '2b', '3b'],
        'col#': ['_', '_', '_'],
        'col#.e': ['_e', '_e', '_e'],
    })
    pd.testing.assert_frame_equal(transformed, expected)

    reverse_transformed = ht.reverse_transform(transformed)
    pd.testing.assert_frame_equal(reverse_transformed, data)


def test_hyper_transformer_field_transformer_correctly_set():
    """Test field_transformers is correctly set through various methods."""
    # Setup
    ht = HyperTransformer()
    data = pd.DataFrame({'col': ['a', 'b', 'c']})

    # getting detected transformers should give the actual transformer
    ht.detect_initial_config(data)
    transformer = ht.get_config()['transformers']['col']
    transformer.new_attribute = 'abc'
    assert ht.get_config()['transformers']['col'].new_attribute == 'abc'

    # the actual transformer should be fitted
    ht.fit(data)
    transformer.new_attribute2 = '123'
    assert ht.get_config()['transformers']['col'].new_attribute == 'abc'
    assert ht.get_config()['transformers']['col'].new_attribute2 == '123'

    # if a transformer was set, it should use the provided instance
    fe = FrequencyEncoder()
    ht.set_config({'sdtypes': {'col': 'categorical'}, 'transformers': {'col': fe}})
    ht.fit(data)
    transformer = ht.get_config()['transformers']['col']
    assert transformer is fe

    # the three cases below make sure any form of acess to the field_transformers
    # correctly accesses and stores the actual transformers
    fe = FrequencyEncoder()
    ht.update_transformers({'col': fe})
    ht.fit(data)
    transformer = ht.get_config()['transformers']['col']
    assert transformer is fe

    ht.update_transformers_by_sdtype('categorical', transformer_name='FrequencyEncoder')
    transformer = ht.get_config()['transformers']['col']
    transformer.new_attribute3 = 'abc'
    ht.fit(data)
    assert ht.get_config()['transformers']['col'].new_attribute3 == 'abc'

    ht.update_sdtypes({'col': 'text'})
    transformer = ht.get_config()['transformers']['col']
    transformer.new_attribute3 = 'abc'
    ht.fit(data)
    assert ht.get_config()['transformers']['col'].new_attribute3 == 'abc'


def _get_hyper_transformer_with_random_transformers(data):
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.update_sdtypes({
        'credit_card': 'pii',
        'name': 'text',
        'signup_day': 'datetime'
    })
    ht.update_transformers({
        'credit_card': AnonymizedFaker('credit_card', 'credit_card_number'),
        'balance': ClusterBasedNormalizer(max_clusters=3),
        'name': RegexGenerator()
    })
    ht.update_transformers_by_sdtype(
        'categorical',
        transformer_name='FrequencyEncoder',
        transformer_parameters={'add_noise': True}
    )

    return ht


def test_hyper_transformer_reset_randomization():
    """Test that the random seeds are properly set and reset.

    If two separate ``HyperTransformer``s are fit, they should have the same parameters
    and produce the same data when transforming. Successive transforming calls should
    yield different results.

    For reverse transforming, different values should be seen if randomization is involved
    unless ``reset_randomization`` is called.
    """
    # Setup
    data = pd.DataFrame({
        'credit_card': ['123456789', '987654321', '192837645', '918273465', '123789456'],
        'age': [18, 25, 54, 60, 31],
        'name': ['Bob', 'Jane', 'Jack', 'Jill', 'Joe'],
        'signup_day': ['1/1/2020', np.nan, '4/1/2019', '12/1/2008', '5/16/2016'],
        'balance': [250, 5400, 150000, np.nan, 91000],
        'card_type': ['Visa', 'Visa', 'Master Card', 'Amex', 'Visa']
    })
    ht1 = _get_hyper_transformer_with_random_transformers(data)
    ht2 = _get_hyper_transformer_with_random_transformers(data)

    # Test transforming multiple times with different transformers
    expected_first_transformed = pd.DataFrame({
        'age': [18.0, 25.0, 54.0, 60.0, 31.0],
        'signup_day': [1.577837e+18, 1.455840e+18, 1.554077e+18, 1.228090e+18, 1.463357e+18],
        'balance.normalized': [
            -2.693016e-01,
            -2.467182e-01,
            3.873711e-01,
            9.571797e-17,
            1.286486e-01
        ],
        'balance.component': [0.0, 0, 0, 0, 0],
        'card_type': [
            0.3273532539594452,
            0.36028672438848,
            0.665212975367718,
            0.8806283675768456,
            0.23386325679767678
        ]
    })
    expected_second_transformed = pd.DataFrame({
        'age': [18.0, 25.0, 54.0, 60.0, 31.0],
        'signup_day': [1.577837e+18, 1.455840e+18, 1.554077e+18, 1.228090e+18, 1.463357e+18],
        'balance.normalized': [
            -2.693016e-01,
            -2.467182e-01,
            3.873711e-01,
            9.571797e-17,
            1.286486e-01
        ],
        'balance.component': [0.0, 0, 0, 0, 0],
        'card_type': [
            0.24748494176070168,
            0.3886965582883772,
            0.7408364277428201,
            0.9194352110397322,
            0.2293601452128275
        ]
    })

    ht1.fit(data)
    first_transformed1 = ht1.transform(data)
    ht2.fit(data)
    first_transformed2 = ht2.transform(data)
    second_transformed1 = ht1.transform(data)

    pd.testing.assert_frame_equal(first_transformed1, expected_first_transformed)
    pd.testing.assert_frame_equal(first_transformed2, expected_first_transformed)
    pd.testing.assert_frame_equal(second_transformed1, expected_second_transformed)

    # test reverse transforming multiple times with different tranformers
    expected_first_reverse = pd.DataFrame({
        'credit_card': [
            '3564483245479407',
            '4863061245886958069',
            '4039466324278480',
            '4217004814656859',
            '4343691397776091'
        ],
        'age': [18, 25, 54, 60, 31],
        'name': ['AAAAA', 'AAAAB', 'AAAAC', 'AAAAD', 'AAAAE'],
        'signup_day': [np.nan, '02/19/2016', '04/01/2019', np.nan, '05/16/2016'],
        'balance': [np.nan, 5400, 150000, np.nan, 91000],
        'card_type': ['Visa', 'Visa', 'Master Card', 'Amex', 'Visa']
    })
    expected_second_reverse = pd.DataFrame({
        'credit_card': [
            '4208002654643',
            '3547584322792794',
            '30187802217181',
            '4138954513622487900',
            '346502559595986'
        ],
        'age': [18, 25, 54, 60, 31],
        'name': ['AAAAF', 'AAAAG', 'AAAAH', 'AAAAI', 'AAAAJ'],
        'signup_day': ['01/01/2020', '02/19/2016', np.nan, '12/01/2008', '05/16/2016'],
        'balance': [250, 5400, np.nan, 61662.5, 91000],
        'card_type': ['Visa', 'Visa', 'Master Card', 'Amex', 'Visa']
    })
    first_reverse1 = ht1.reverse_transform(first_transformed1)
    first_reverse2 = ht2.reverse_transform(first_transformed1)
    second_reverse1 = ht1.reverse_transform(first_transformed1)
    pd.testing.assert_frame_equal(first_reverse1, expected_first_reverse)
    pd.testing.assert_frame_equal(first_reverse2, expected_first_reverse)
    pd.testing.assert_frame_equal(expected_second_reverse, second_reverse1)

    # Test resetting randomization
    ht1.reset_randomization()

    transformed_post_reset = ht1.reverse_transform(first_transformed1)
    pd.testing.assert_frame_equal(transformed_post_reset, expected_first_reverse)


def test_hyper_transformer_cluster_based_normalizer_randomization():
    """Test that the ``ClusterBasedNormalizer`` handles randomization correctly.

    If the ``ClusterBasedNormalizer`` transforms the same data multiple times,
    it may yield different results due to randomness. However, if a new instance is created,
    each matching call should yield the same results (ie. call 1 of the first transformer
    should match call 1 of the second).
    """
    data = get_demo(100)
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.update_transformers({
        'age': ClusterBasedNormalizer()
    })
    ht.fit(data)
    transformed1 = ht.transform(data)
    transformed2 = ht.transform(data)

    assert any(transformed1['age.normalized'] != transformed2['age.normalized'])

    ht2 = HyperTransformer()
    ht2.detect_initial_config(data)
    ht2.update_transformers({
        'age': ClusterBasedNormalizer()
    })
    ht2.fit(data)

    pd.testing.assert_frame_equal(transformed1, ht2.transform(data))


def test_hypertransformer_anonymized_faker():
    """Test ``AnonymizedFaker`` generates different random values for different columns.

    Issue: https://github.com/sdv-dev/RDT/issues/619.
    """
    # Setup
    data = pd.DataFrame({
        'id1': ['a', 'b', 'c'],
        'id2': ['d', 'e', 'f'],
    })
    ht = HyperTransformer()

    # Run - simple run
    ht.detect_initial_config(data)
    ht.update_sdtypes({
        'id1': 'pii',
        'id2': 'pii'
    })
    ht.update_transformers({
        'id1': AnonymizedFaker(),
        'id2': AnonymizedFaker()
    })
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed1 = ht.reverse_transform(transformed)

    # Assert
    assert reverse_transformed1['id1'].tolist() != reverse_transformed1['id2'].tolist()

    # Run - make sure transforming again returns different values than the original transform
    transformed = ht.transform(data)
    reverse_transformed2 = ht.reverse_transform(transformed)

    # Assert
    assert reverse_transformed2['id1'].tolist() != reverse_transformed2['id2'].tolist()
    assert reverse_transformed1['id1'].tolist() != reverse_transformed2['id1'].tolist()
    assert reverse_transformed1['id2'].tolist() != reverse_transformed2['id2'].tolist()

    # Run - make sure resetting randomization works
    ht.reset_randomization()
    transformed = ht.transform(data)
    reverse_transformed3 = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(reverse_transformed1, reverse_transformed3)


def test_hypertransformer_pseudo_anonymized_faker():
    """Test ``PseudoAnonymizedFaker`` generates different random values for different columns."""
    # Setup
    data = pd.DataFrame({
        'id1': ['a', 'b', 'c'],
        'id2': ['a', 'b', 'c'],
    })
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.update_sdtypes({
        'id1': 'pii',
        'id2': 'pii'
    })
    ht.update_transformers({
        'id1': PseudoAnonymizedFaker(),
        'id2': PseudoAnonymizedFaker()
    })
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed1 = ht.reverse_transform(transformed)

    # Assert
    assert reverse_transformed1['id1'].tolist() != reverse_transformed1['id2'].tolist()

    # Run - run it again on the exact same data
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.update_sdtypes({
        'id1': 'pii',
        'id2': 'pii'
    })
    ht.update_transformers({
        'id1': PseudoAnonymizedFaker(),
        'id2': PseudoAnonymizedFaker()
    })
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed2 = ht.reverse_transform(transformed)

    # Assert - different instances of the same transformer should return the same result
    assert reverse_transformed1['id1'].tolist() == reverse_transformed2['id1'].tolist()


def test_hypertransformer_anonymized_faker_multi_table():
    """Test ``AnonymizedFaker`` generates different values for columns with same name."""
    # Setup
    data1 = pd.DataFrame({
        'id1': ['a', 'b', 'c'],
        'id2': ['a', 'b', 'c'],
    })
    data2 = pd.DataFrame({
        'id1': ['d', 'e', 'f'],
        'id2': ['d', 'e', 'f'],
    })
    ht = HyperTransformer()

    # Run on data1
    ht.detect_initial_config(data1)
    ht.update_sdtypes({
        'id1': 'pii',
        'id2': 'pii'
    })
    ht.update_transformers({
        'id1': AnonymizedFaker(),
        'id2': PseudoAnonymizedFaker()
    })
    ht.fit(data1)
    transformed = ht.transform(data1)
    reverse_transformed1 = ht.reverse_transform(transformed)

    # Run on data2
    ht.detect_initial_config(data2)
    ht.update_sdtypes({
        'id1': 'pii',
        'id2': 'pii'
    })
    ht.update_transformers({
        'id1': AnonymizedFaker(),
        'id2': PseudoAnonymizedFaker()
    })
    ht.fit(data2)
    transformed = ht.transform(data2)
    reverse_transformed2 = ht.reverse_transform(transformed)

    # Assert
    assert reverse_transformed1['id1'].tolist() != reverse_transformed2['id1'].tolist()
    assert reverse_transformed1['id2'].tolist() != reverse_transformed2['id2'].tolist()
