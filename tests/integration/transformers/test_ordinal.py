import warnings

import numpy as np
import pandas as pd
import pytest

from rdt.hyper_transformer import HyperTransformer
from rdt.transformers import (
    OrderedLabelEncoder,
    OrderedUniformEncoder,
)


class TestOrderedUniformEncoder:
    """Test class for the OrderedUniformEncoder."""

    def test_order(self):
        """Test that the ``order`` parameter is respected."""
        # Setup
        data = pd.DataFrame({'column_name': [1, 2, 3, 2, np.nan, 1, 1]})
        transformer = OrderedUniformEncoder(order=[2, 3, np.nan, 1])
        column = 'column_name'

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)
        expected_order = pd.Series([2, 3, np.nan, 1], dtype=object)

        # Asserts
        pd.testing.assert_series_equal(reverse[column], data[column])
        pd.testing.assert_series_equal(transformer.order, expected_order)

    def test_string(self):
        """Test that the transformer works with string labels."""
        # Setup
        data = pd.DataFrame({'column_name': ['b', 'a', 'c', 'a', np.nan, 'b', 'b']})
        transformer = OrderedUniformEncoder(order=['a', 'c', np.nan, 'b'])
        column = 'column_name'

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

        # Asserts
        pd.testing.assert_series_equal(reverse[column], data[column])

    def test_mixed_dtypes(self):
        """Test that the transformer works with mixture of dtypes labels."""
        # Setup
        data = pd.DataFrame({'column_name': [1, 'a', 'c', 'a', np.nan, 1, 1]})
        transformer = OrderedUniformEncoder(order=['a', 'c', np.nan, 1])
        column = 'column_name'

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

        # Asserts
        pd.testing.assert_series_equal(reverse[column], data[column])


def test_ordered_label_encoder():
    """Test the OrderedLabelEncoder end to end.

    Input:
        - pandas.DataFrame of different types of data.

    Output:
        - Transformed data should have values based on the provided order.
        - Reverse transformed data should match the input
    """

    data = pd.DataFrame(['two', 3, 1, np.nan, 'zero'], columns=['column_name'])
    transformer = OrderedLabelEncoder(order=['zero', 1, 'two', 3, np.nan])
    transformer.fit(data, 'column_name')

    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    expected = pd.DataFrame([2, 3, 1, 4, 0], columns=['column_name'])
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_frame_equal(reverse, data)


def test_ordered_label_encoder_nans():
    """The the OrderedLabelEncoder with missing values.

    Input:
        - pandas.DataFrame of different types of data and different types of missing values.

    Output:
        - Transformed data should have values based on the provided order.
        - Reverse transformed data should match the input
    """

    data = pd.DataFrame(['two', 3, 1, np.nan, 'zero', None], columns=['column_name'])
    transformer = OrderedLabelEncoder(order=['zero', 1, 'two', 3, None])
    transformer.fit(data, 'column_name')

    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    expected = pd.DataFrame([2, 3, 1, 4, 0, 4], columns=['column_name'])
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_frame_equal(reverse, data)


def test_ordered_label_encoder_numerical_nans_no_warning():
    """Ensure OrderedLabelEncoder does not emit FutureWarning with nan values.

    Related to Issue #793 (https://github.com/sdv-dev/RDT/issues/793)
    """
    # Setup
    data = pd.DataFrame({'column_name': pd.Series([1, 2, float('nan'), np.nan], dtype='object')})
    column = 'column_name'

    # Run and Assert
    transformer = OrderedLabelEncoder(order=[1, 2, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_ordered_label_encoder_default_order_by_numerical():
    """Test the OrderedLabelEncoder orders alphanumerically if `order` is None."""

    data = pd.DataFrame([5, np.nan, 3.11, 100, 67.8, -2.5], columns=['column_name'])

    transformer = OrderedLabelEncoder(order=None)
    transformer.fit(data, 'column_name')
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    expected = pd.DataFrame([2, 5, 1, 4, 3, 0], columns=['column_name'])
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_frame_equal(reverse, data)


def test_ordered_label_encoder_order_by_alphabetical():
    """Test the OrderedLabelEncoder orders alphanumerically if `order` is None."""
    data = pd.DataFrame(['one', 'two', np.nan, 'three', 'four'], columns=['column_name'])

    transformer = OrderedLabelEncoder(order=None)
    transformer.fit(data, 'column_name')
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    expected = pd.DataFrame([1, 3, 4, 2, 0], columns=['column_name'])
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_frame_equal(reverse, data)


ordinal_transformers = [
    OrderedUniformEncoder(order=[1, 'two', 3, 'four'], missing_value_encoding=None),
    OrderedLabelEncoder(order=[1, 'two', 3, 'four'], missing_value_encoding=None),
]


@pytest.mark.parametrize('transformer', ordinal_transformers)
def test_ordinal_transformers_missing_value_encoding_none(transformer):
    """Test categorical transformers preserve missing values when configured to do so."""
    # Setup
    data = pd.DataFrame({'col': [1, None, 'two', np.nan, 3, 'four']})

    # Run
    transformer.fit(data, 'col')
    transformed = transformer.transform(data)
    reverse_transformed = transformer.reverse_transform(transformed)

    # Assert
    expected_missing_values = pd.Series([False, True, False, True, False, False], name='col')
    pd.testing.assert_series_equal(transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(reverse_transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(
        reverse_transformed.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        data.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize('transformer', ordinal_transformers)
def test_ordinal_transformers_missing_value_encoding_none_when_nulls_not_seen(transformer):
    """Test missing values remain missing when they were not seen during fitting."""
    # Setup
    fit_data = pd.DataFrame({'col': [1, 'two', 3, 'four']})
    transform_data = pd.DataFrame({'col': [1, None, 'two', np.nan, 3, 'four']})

    # Run
    transformer.fit(fit_data, 'col')
    with warnings.catch_warnings(record=True) as recorded_warnings:
        transformed = transformer.transform(transform_data)

    reverse_transformed = transformer.reverse_transform(transformed)

    # Assert
    assert len(recorded_warnings) == 0
    expected_missing_values = pd.Series([False, True, False, True, False, False], name='col')
    pd.testing.assert_series_equal(transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(reverse_transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(
        reverse_transformed.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        transform_data.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize('transformer', ordinal_transformers)
def test_ordinal_transformers_missing_value_encoding_none_all_missing(transformer):
    """Test transformers with no learned categories keep missing values missing."""
    # Setup
    data = pd.DataFrame({'col': pd.Series([None, np.nan, pd.NA], dtype='object')})

    # Run
    transformer.fit(data, 'col')
    transformed = transformer.transform(data)
    reverse_transformed = transformer.reverse_transform(transformed)

    # Assert
    assert transformed['col'].isna().all()
    assert reverse_transformed['col'].isna().all()
    assert reverse_transformed.shape == data.shape


@pytest.mark.parametrize('transformer', ordinal_transformers)
def test_ordinal_transformers_missing_value_encoding_none_reverse_passes_nulls_through(
    transformer,
):
    """Test missing values produced downstream stay missing during reverse transform."""
    # Setup
    data = pd.DataFrame({'col': [1, 'two', 3, 'four']})

    # Run
    transformer.fit(data, 'col')
    transformed = transformer.transform(data)
    transformed.loc[[1, 3], 'col'] = np.nan
    reverse_transformed = transformer.reverse_transform(transformed)

    # Assert
    expected_missing_values = pd.Series([False, True, False, True], name='col')
    pd.testing.assert_series_equal(reverse_transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(
        reverse_transformed.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        data.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        check_dtype=False,
    )


def test_ordered_label_encoders_missing_value_encoding_none_with_category_dtype():
    """Test label encoders keep transformed pandas category columns numeric."""
    # Setup
    data = pd.DataFrame({'col': pd.Series(['a', None, 'b'], dtype='category')})
    transformer = OrderedLabelEncoder(order=['a', 'b'], missing_value_encoding=None)

    # Run
    transformer.fit(data, 'col')
    transformed = transformer.transform(data)
    reverse_transformed = transformer.reverse_transform(transformed)

    # Assert
    assert pd.api.types.is_numeric_dtype(transformed['col'])
    expected_missing_values = pd.Series([False, True, False], name='col')
    pd.testing.assert_series_equal(transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(reverse_transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(
        reverse_transformed.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        data.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    'transformer',
    [
        OrderedUniformEncoder(order=[1, 'two', 3, 'four', None]),
        OrderedLabelEncoder(order=[1, 'two', 3, 'four', None]),
    ],
)
def test_ordinal_transformers_default_missing_value_encoding_new_category(transformer):
    """Test default missing value handling continues to encode missing as a category."""
    # Setup
    data = pd.DataFrame({'col': [1, None, 'two', np.nan, 3, 'four']})

    # Run
    transformer.fit(data, 'col')
    transformed = transformer.transform(data)
    reverse_transformed = transformer.reverse_transform(transformed)

    # Assert
    assert transformed['col'].notna().all()
    expected_missing_values = pd.Series([False, True, False, True, False, False], name='col')
    pd.testing.assert_series_equal(reverse_transformed['col'].isna(), expected_missing_values)
    pd.testing.assert_series_equal(
        reverse_transformed.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        data.loc[~expected_missing_values, 'col'].reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize('sdtype', ['id', 'text'])
@pytest.mark.parametrize('transformer', ordinal_transformers)
def test_ordinal_transformers_with_id_sdtype(sdtype, transformer):
    # Setup
    data = pd.DataFrame({
        'col': [1, 'two', 3, 'four', None],
    })
    hyper_transformer = HyperTransformer()
    config = {'sdtypes': {'col': sdtype}, 'transformers': {'col': transformer}}

    # Run
    hyper_transformer.set_config(config)
    hyper_transformer.fit(data)
    transformed = hyper_transformer.transform(data)
    reverse_transformed = hyper_transformer.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(data, reverse_transformed)
