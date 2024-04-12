import pickle
import warnings
from io import BytesIO

import numpy as np
import pandas as pd

from rdt.transformers import (
    FrequencyEncoder, LabelEncoder, OneHotEncoder, OrderedLabelEncoder, OrderedUniformEncoder,
    UniformEncoder)


class TestUniformEncoder:
    """Test class for the UniformEncoder."""

    def test_frequency(self):
        """Test that the frequencies sum up to 1.0."""
        # Setup
        data_1 = pd.DataFrame({'column_name': [1, 2, 3, 2, 1, 1, 1]})
        data_2 = pd.DataFrame({'column_name': [1, 2, 3]})
        column = 'column_name'

        transformer_1 = UniformEncoder()
        transformer_2 = UniformEncoder()

        # Run
        transformer_1.fit(data_1, column)
        transformer_2.fit(data_2, column)

        # Asserts
        assert sum(transformer_1.frequencies.values()) == 1.0
        assert sum(transformer_2.frequencies.values()) == 1.0

    def test_clip_interval(self):
        """Test that the first interval starts at 0 and last ends at 0."""
        # Setup
        data = pd.DataFrame({'column_name': [1, 2, 3, 2, 1, 1, 1]})
        column = 'column_name'
        transformer = UniformEncoder(order_by='numerical_value')

        # Run
        transformer.fit(data, column)

        # Asserts
        assert transformer.intervals[1][0] == 0.0
        assert transformer.intervals[3][-1] == 1.0

    def test__reverse_transform(self):
        """Test the ``reverse_transform``."""
        # Setup
        data = pd.DataFrame({'column_name': [1, 2, 3, 2, 2, 1, 3, 3, 2]})
        column = 'column_name'

        transformer = UniformEncoder()

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        output = transformer.reverse_transform(transformed)

        # Asserts
        pd.testing.assert_series_equal(output['column_name'], data['column_name'])

    def test__reverse_transform_negative_transformed_values(self):
        """Test the ``reverse_transform``."""
        # Setup
        data = pd.DataFrame({'column_name': [1, 2, 3, 2, 2, 1, 3, 3, 2]})
        column = 'column_name'
        transformer = UniformEncoder()

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        transformed.loc[1, 'column_name'] = -0.1
        transformed.loc[2, 'column_name'] = 100
        output = transformer.reverse_transform(transformed)

        # Asserts
        # Make sure there is no Nan values due to the negative number or large upper bound number
        assert not any(pd.isna(output).to_numpy())

    def test__reverse_transform_nans(self):
        """Test ``reverse_transform`` for data with NaNs."""
        # Setup
        data = pd.DataFrame({
            'column_name': ['a', 'b', 'c', np.nan, 'c', 'b', 'b', 'a', 'b', np.nan]
        })
        column = 'column_name'

        transformer = UniformEncoder()

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        output = transformer.reverse_transform(transformed)

        # Asserts
        pd.testing.assert_series_equal(output[column], data[column])

    def test_uniform_encoder_unseen_transform_nan(self):
        """Ensure UniformEncoder works when np.nan to transform wasn't seen during fit."""
        # Setup
        fit_data = pd.DataFrame([1.0, 2.0, 3.0], columns=['column_name'])
        transform_data = pd.DataFrame([1, 2, 3, np.nan], columns=['column_name'])
        column = 'column_name'

        transformer = UniformEncoder()

        # Run
        transformer.fit(fit_data, column)
        transformed = transformer.transform(transform_data)
        reverse = transformer.reverse_transform(transformed)

        # Asserts
        pd.testing.assert_frame_equal(reverse[:3], transform_data[:3])
        assert reverse.iloc[3][0] in {1, 2, 3}

    def test_transform_with_nans(self):
        """Test the ``UniformEncoder`` works properly with nan values."""
        # Setup
        data = pd.DataFrame({
            'bool': [True, False, None, False, True],
            'mycol': ['a', 'b', 'a', None, np.nan],
        })
        ue = UniformEncoder()

        # Run
        ue.fit(data, 'mycol')
        transformed = ue.transform(data)
        out = ue.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(out, data)


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

    def test_mixt_dtype(self):
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


def test_frequency_encoder_numerical_nans():
    """Ensure FrequencyEncoder works on numerical + nan only columns."""

    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])
    column = 'column_name'

    transformer = FrequencyEncoder()
    transformer.fit(data, column)
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_frequency_encoder_unseen_transform_data():
    """Ensure FrequencyEncoder works when data to transform wasn't seen during fit."""

    fit_data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])
    transform_data = pd.DataFrame([1, 2, np.nan, 3], columns=['column_name'])
    column = 'column_name'

    transformer = FrequencyEncoder()
    transformer.fit(fit_data, column)
    transformed = transformer.transform(transform_data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse[:3], transform_data[:3])
    assert reverse.iloc[3][0] in {1, 2} or pd.isna(reverse.iloc[3])[0]


def test_frequency_encoder_unseen_transform_nan():
    """Ensure FrequencyEncoder works when np.nan to transform wasn't seen during fit."""

    fit_data = pd.DataFrame([1.0, 2.0, 3.0], columns=['column_name'])
    transform_data = pd.DataFrame([1, 2, 3, np.nan], columns=['column_name'])
    column = 'column_name'

    transformer = FrequencyEncoder()
    transformer.fit(fit_data, column)
    transformed = transformer.transform(transform_data)
    reverse = transformer.reverse_transform(transformed)
    pd.testing.assert_frame_equal(reverse[:3], transform_data[:3])
    assert reverse.iloc[3][0] in {1, 2, 3}


def test_frequency_encoder_pickle_nans():
    """Ensure that FrequencyEncoder can be pickled and loaded with nan value."""
    # setup
    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])
    column = 'column_name'

    transformer = FrequencyEncoder()
    transformer.fit(data, column)
    transformed = transformer.transform(data)

    # create pickle file on memory
    bytes_io = BytesIO()
    pickle.dump(transformer, bytes_io)
    # rewind
    bytes_io.seek(0)

    # run
    pickled_transformer = pickle.load(bytes_io)

    # assert
    pickle_transformed = pickled_transformer.transform(data)
    pd.testing.assert_frame_equal(pickle_transformed, transformed)


def test_frequency_encoder_strings():
    """Test the FrequencyEncoder on string data.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform on string data. Expect that the reverse transformed data
    is the same as the input.

    Input:
        - 4 rows of string data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame(['a', 'b', 'a', 'c'], columns=['column_name'])
    column = 'column_name'
    transformer = FrequencyEncoder()

    # run
    transformer.fit(data, column)
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_frequency_encoder_strings_2_categories():
    """Test the FrequencyEncoder on string data.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform on string data, when there are 2 categories of strings with
    the same value counts. Expect that the reverse transformed data is the
    same as the input.

    Input:
        - 4 rows of string data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame(['a', 'b', 'a', 'b'], columns=['column_name'])
    column = 'column_name'
    transformer = FrequencyEncoder()

    transformer.fit(data, column)
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_frequency_encoder_integers():
    """Test the FrequencyEncoder on integer data.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform on integer data. Expect that the reverse transformed data is the
    same as the input.

    Input:
        - 4 rows of int data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([1, 2, 3, 2], columns=['column_name'])
    column = 'column_name'
    transformer = FrequencyEncoder()

    # run
    transformer.fit(data, column)
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_frequency_encoder_bool():
    """Test the FrequencyEncoder on boolean data.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform on boolean data. Expect that the reverse transformed data is the
    same as the input.

    Input:
        - 4 rows of bool data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, False, True, False], columns=['column_name'])
    column = 'column_name'
    transformer = FrequencyEncoder()

    # run
    transformer.fit(data, column)
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_frequency_encoder_mixed():
    """Test the FrequencyEncoder on mixed type data.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform on mixed type data. Expect that the reverse transformed data is
    the same as the input.

    Input:
        - 4 rows of mixed data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, 'a', 1, None], columns=['column_name'])
    column = 'column_name'
    transformer = FrequencyEncoder()

    # run
    transformer.fit(data, column)
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_frequency_encoder_mixed_more_rows():
    """Test the FrequencyEncoder on mixed type data.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform on mixed type data, when there is a larger number of rows.
    Expect that the reverse transformed data is the same as the input.

    Input:
        - 4 rows of mixed data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, 'a', 1, None], columns=['column_name'])
    column = 'column_name'
    transform_data = pd.DataFrame(['a', 1, None, 'a', True, 1], columns=['column_name'])
    transformer = FrequencyEncoder()

    # run
    transformer.fit(data, column)
    transformed = transformer.transform(transform_data)
    reverse = transformer.reverse_transform(transformed)

    # assert
    pd.testing.assert_frame_equal(transform_data, reverse)


def test_frequency_encoder_noise():
    """Test the FrequencyEncoder with ``add_noise``.

    Ensure that the FrequencyEncoder can fit, transform, and reverse
    transform when ``add_noise = True``.

    Input:
        - Many rows of int data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame(np.random.choice(a=range(100), size=10000), columns=['column_name'])
    column = 'column_name'
    transformer = FrequencyEncoder(add_noise=True)

    # run
    transformer.fit(data, column)
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_one_hot_numerical_nans():
    """Ensure OneHotEncoder works on numerical + nan only columns."""

    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])
    column = 'column_name'

    transformer = OneHotEncoder()
    transformer.fit(data, column)
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_one_hot_doesnt_warn(tmp_path):
    """Ensure OneHotEncoder doesn't warn when saving and loading GH#616."""
    # Setup
    data = pd.DataFrame({'column_name': [1.0, 2.0, np.nan, 2.0, 3.0, np.nan, 3.0]})
    ohe = OneHotEncoder()

    # Run
    ohe.fit(data, 'column_name')
    tmp = tmp_path / 'ohe.pkl'
    with open(tmp, 'wb') as f:
        pickle.dump(ohe, f)
    with open(tmp, 'rb') as f:
        ohe_loaded = pickle.load(f)

    # Assert
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ohe_loaded.transform(data)


def test_one_hot_categoricals():
    """Ensure OneHotEncoder works on categorical data. GH#751"""
    # Setup
    test_data = pd.DataFrame(data={
        'A': ['Yes', 'No', 'Yes', 'Maybe', 'No']
    })
    test_data['A'] = test_data['A'].astype('category')
    transformer = OneHotEncoder()

    # Run
    transformed_data = transformer.fit_transform(test_data, column='A')

    # Assert
    pd.testing.assert_frame_equal(
        transformed_data,
        pd.DataFrame({
            'A.value0': [1, 0, 1, 0, 0],
            'A.value1': [0, 1, 0, 0, 1],
            'A.value2': [0, 0, 0, 1, 0],
        }),
        check_dtype=False
    )

    # Run
    reversed_data = transformer.reverse_transform(transformed_data)

    # Assert
    pd.testing.assert_frame_equal(reversed_data, test_data)


def test_label_numerical_2d_array():
    """Ensure LabelEncoder works on numerical + nan only columns."""

    data = pd.DataFrame(['a', 'b', 'c', 'd'], columns=['column_name'])
    column = 'column_name'

    transformer = LabelEncoder()
    transformer.fit(data, column)

    transformed = pd.DataFrame([0., 1., 2., 3.], columns=['column_name'])
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_label_numerical_nans():
    """Ensure LabelEncoder works on numerical + nan only columns."""

    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])
    column = 'column_name'

    transformer = LabelEncoder()
    transformer.fit(data, column)
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_label_encoder_order_by_numerical():
    """Test the LabelEncoder appropriately transforms data if `order_by` is 'numerical_value'.

    Input:
        - pandas.DataFrame of numeric data.

    Output:
        - Transformed data should map labels to values based on numerical order.
    """

    data = pd.DataFrame([5, np.nan, 3.11, 100, 67.8, -2.5], columns=['column_name'])

    transformer = LabelEncoder(order_by='numerical_value')
    transformer.fit(data, 'column_name')
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    expected = pd.DataFrame([2, 5, 1, 4, 3, 0], columns=['column_name'])
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_frame_equal(reverse, data)


def test_label_encoder_order_by_alphabetical():
    """Test the LabelEncoder appropriately transforms data if `order_by` is 'alphabetical'.

    Input:
        - pandas.DataFrame of string data.

    Output:
        - Transformed data should map labels to values based on alphabetical order.
    """

    data = pd.DataFrame(['one', 'two', np.nan, 'three', 'four'], columns=['column_name'])

    transformer = LabelEncoder(order_by='alphabetical')
    transformer.fit(data, 'column_name')
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    expected = pd.DataFrame([1, 3, 4, 2, 0], columns=['column_name'])
    pd.testing.assert_frame_equal(transformed, expected)
    pd.testing.assert_frame_equal(reverse, data)


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
