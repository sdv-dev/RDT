import numpy as np
import pandas as pd

from rdt.transformers.text import RegexGenerator


def test_regexgenerator():
    """Test ``RegexGenerator`` with the default parameters."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e']
    })

    instance = RegexGenerator()
    transformed = instance.fit_transform(data, 'id')
    reverse_transform = instance.reverse_transform(transformed)

    expected_transformed = pd.DataFrame({
        'username': ['a', 'b', 'c', 'd', 'e']
    })
    expected_reverse_transformed = pd.DataFrame({
        'username': ['a', 'b', 'c', 'd', 'e'],
        'id': ['AAAAA', 'AAAAB', 'AAAAC', 'AAAAD', 'AAAAE'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_with_custom_regex():
    """Test the ``RegexGenerator`` with a custom regex format."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e'],
    })

    instance = RegexGenerator(regex_format='[1-9]')
    transformed = instance.fit_transform(data, 'id')
    reverse_transform = instance.reverse_transform(transformed)

    expected_transformed = pd.DataFrame({
        'username': ['a', 'b', 'c', 'd', 'e'],
    })

    expected_reverse_transformed = pd.DataFrame({
        'username': ['a', 'b', 'c', 'd', 'e'],
        'id': ['1', '2', '3', '4', '5'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_with_nans():
    """Test the ``RegexGenerator`` with a custom regex format and ``nans``."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'c', 'd', 'e']
    })

    instance = RegexGenerator('[A-Z]', model_missing_values=True)
    transformed = instance.fit_transform(data, 'username')
    reverse_transform = instance.reverse_transform(transformed)

    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username.is_null': [0.0, 1.0, 0.0, 0.0, 0.0]
    })

    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['A', np.nan, 'C', 'D', 'E'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_data_length_bigger_than_regex():
    """Test the ``RegexGenerator`` with short regex and more data length."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'c', 'd', 'e']
    })

    instance = RegexGenerator('[a-b]', model_missing_values=True)
    transformed = instance.fit_transform(data, 'username')
    reverse_transform = instance.reverse_transform(transformed)

    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username.is_null': [0.0, 1.0, 0.0, 0.0, 0.0]
    })

    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'a', 'b', 'a'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_input_data_bigger_than_data_length():
    """Test the ``RegexGenerator`` with input dataframe bigger than the learned data length."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e']
    })

    instance = RegexGenerator('[a-b]', model_missing_values=True)
    instance.fit(data, 'username')

    transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    reverse_transform = instance.reverse_transform(transformed)

    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'username': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
    })

    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)
