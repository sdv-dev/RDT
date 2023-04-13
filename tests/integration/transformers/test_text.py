import pickle

import numpy as np
import pandas as pd

from rdt.transformers.text import RegexGenerator


def test_regexgenerator():
    """Test ``RegexGenerator`` with the default parameters."""
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e']
    })

    # Run
    instance = RegexGenerator()
    transformed = instance.fit_transform(data, 'id')
    reverse_transform = instance.reverse_transform(transformed)

    # Assert
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
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e'],
    })

    # Run
    instance = RegexGenerator(regex_format='[1-9]')
    transformed = instance.fit_transform(data, 'id')
    reverse_transform = instance.reverse_transform(transformed)

    # Assert
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
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'c', 'd', 'e']
    })

    # Run
    instance = RegexGenerator('[A-Z]')
    transformed = instance.fit_transform(data, 'username')
    reverse_transform = instance.reverse_transform(transformed)

    # Assert
    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
    })

    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['A', 'B', 'C', 'D', 'E'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_data_length_bigger_than_regex():
    """Test the ``RegexGenerator`` with short regex and more data length."""
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'c', 'd', 'e']
    })

    # Run
    instance = RegexGenerator('[a-b]')
    transformed = instance.fit_transform(data, 'username')
    reverse_transform = instance.reverse_transform(transformed)

    # Assert
    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
    })

    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'a', 'b', 'a'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_input_data_bigger_than_data_length():
    """Test the ``RegexGenerator`` with input dataframe bigger than the learned data length."""
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e']
    })

    # Run
    instance = RegexGenerator('[a-b]')
    instance.fit(data, 'username')

    transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    reverse_transform = instance.reverse_transform(transformed)

    # Assert
    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'username': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
    })

    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)


def test_regexgenerator_called_multiple_times():
    """Test the ``RegexGenerator`` with short regex and called multiple times.

    This test ensures that when ``enforce_uniqueness`` is ``False`` this generator will continue
    to work.
    """
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'c', 'd', 'e']
    })

    instance = RegexGenerator('[a-c]')

    # Transform
    transformed = instance.fit_transform(data, 'username')

    # Assert Transform
    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
    })
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    # Reverse Transform
    first_reverse_transform = instance.reverse_transform(transformed)

    # Assert Reverse Transform
    expected_reverse_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'a', 'b'],
    })
    pd.testing.assert_frame_equal(first_reverse_transform, expected_reverse_transformed)

    # Reverse Transform Again
    second_reverse_transform = instance.reverse_transform(transformed.head(1))

    # Assert Reverse Transform
    expected_reverse_transformed = pd.DataFrame({
        'id': [1],
        'username': ['a'],
    })
    pd.testing.assert_frame_equal(second_reverse_transform, expected_reverse_transformed)

    # Reverse Transform Again
    third_reverse_transform = instance.reverse_transform(transformed.head(1))

    # Assert Reverse Transform
    expected_reverse_transformed = pd.DataFrame({
        'id': [1],
        'username': ['b'],
    })
    pd.testing.assert_frame_equal(third_reverse_transform, expected_reverse_transformed)


def test_regexgenerator_called_multiple_times_enforce_uniqueness():
    """Test that calling multiple times with ``enforce_uniqueness`` returns unique values."""
    # Setup
    data = pd.DataFrame({'my_column': np.arange(10)})
    generator = RegexGenerator(enforce_uniqueness=True)

    # Run
    transformed_data = generator.fit_transform(data, 'my_column')
    first_reverse_transform = generator.reverse_transform(transformed_data.head(3))
    second_reverse_transform = generator.reverse_transform(transformed_data.head(5))

    # Assert
    expected_first_reverse_transform = pd.DataFrame({
        'my_column': ['AAAAA', 'AAAAB', 'AAAAC']
    })
    expected_second_reverse_transform = pd.DataFrame({
        'my_column': ['AAAAD', 'AAAAE', 'AAAAF', 'AAAAG', 'AAAAH']
    })
    pd.testing.assert_frame_equal(first_reverse_transform, expected_first_reverse_transform)
    pd.testing.assert_frame_equal(second_reverse_transform, expected_second_reverse_transform)


def test_regexgenerator_pickled(tmpdir):
    """Test that ensures that ``RegexGenerator`` can be pickled."""
    # Setup
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e']
    })

    # Run
    instance = RegexGenerator()
    transformed = instance.fit_transform(data, 'id')
    instance.reverse_transform(transformed)

    # Pickle
    with open(tmpdir / 'file.pkl', 'wb') as f:
        pickle.dump(instance, f)

    with open(tmpdir / 'file.pkl', 'rb') as f:
        loaded = pickle.load(f)

    # Assert
    assert next(instance.generator) == next(loaded.generator)


def test_regexgenerator_with_many_possibilities():
    """Test the ``RegexGenerator`` with regex containing many possibilities."""
    # Setup
    data = pd.DataFrame({
        'id': ['a' * 50, 'a' * 49 + 'b', 'a' * 49 + 'c', 'a' * 49 + 'd', 'a' * 49 + 'e'],
        'username': ['aa', 'bb', 'cc', 'dd', 'ee'],
    })

    # Run
    instance = RegexGenerator(regex_format='[a-z]{50}')
    transformed = instance.fit_transform(data, 'id')
    reverse_transform = instance.reverse_transform(transformed)

    # Assert
    expected_transformed = pd.DataFrame({
        'username': ['aa', 'bb', 'cc', 'dd', 'ee'],
    })

    expected_reverse_transformed = pd.DataFrame({
        'username': ['aa', 'bb', 'cc', 'dd', 'ee'],
        'id': ['a' * 50, 'a' * 49 + 'b', 'a' * 49 + 'c', 'a' * 49 + 'd', 'a' * 49 + 'e'],
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)
