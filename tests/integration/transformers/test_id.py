import pickle

import numpy as np
import pandas as pd

from rdt import HyperTransformer, get_demo
from rdt.transformers.id import IndexGenerator, RegexGenerator


class TestIndexGenerator:
    def test_end_to_end(self):
        """End to end test of the ``IndexGenerator``."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
        })

        # Run
        transformer = IndexGenerator(prefix='id_', starting_value=100, suffix='_X')
        transformed = transformer.fit_transform(data, 'id')
        reverse_transform = transformer.reverse_transform(transformed)
        reverse_transform_2 = transformer.reverse_transform(transformed)
        transformer.reset_randomization()
        reverse_transform_3 = transformer.reverse_transform(transformed)

        # Assert
        expected_transformed = pd.DataFrame({'username': ['a', 'b', 'c', 'd', 'e']})

        expected_reverse_transform = pd.DataFrame({
            'username': ['a', 'b', 'c', 'd', 'e'],
            'id': ['id_100_X', 'id_101_X', 'id_102_X', 'id_103_X', 'id_104_X'],
        })

        expected_reverse_transform_2 = pd.DataFrame({
            'username': ['a', 'b', 'c', 'd', 'e'],
            'id': ['id_105_X', 'id_106_X', 'id_107_X', 'id_108_X', 'id_109_X'],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transform)
        pd.testing.assert_frame_equal(reverse_transform_2, expected_reverse_transform_2)
        pd.testing.assert_frame_equal(reverse_transform_3, expected_reverse_transform)


class TestRegexGenerator:
    def test_regexgenerator(self):
        """Test ``RegexGenerator`` with the default parameters."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
        })

        # Run
        instance = RegexGenerator()
        transformed = instance.fit_transform(data, 'id')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        expected_transformed = pd.DataFrame({'username': ['a', 'b', 'c', 'd', 'e']})
        expected_reverse_transformed = pd.DataFrame({
            'username': ['a', 'b', 'c', 'd', 'e'],
            'id': ['AAAAA', 'AAAAB', 'AAAAC', 'AAAAD', 'AAAAE'],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)

    def test_with_custom_regex(self):
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

    def test_with_nans(self):
        """Test the ``RegexGenerator`` with a custom regex format and ``nans``."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', np.nan, 'c', 'd', 'e'],
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

    def test_data_length_bigger_than_regex(self):
        """Test the ``RegexGenerator`` with short regex and more data length."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', np.nan, 'c', 'd', 'e'],
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

    def test_input_data_bigger_than_data_length(self):
        """Test the ``RegexGenerator`` with input dataframe bigger than the learned data length."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
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

    def test_called_multiple_times(self):
        """Test the ``RegexGenerator`` with short regex and called multiple times.

        This test ensures that when ``cardinality_rule`` is ``None`` this generator will
        continue to work.
        """
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', np.nan, 'c', 'd', 'e'],
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

    def test_called_multiple_times_cardinality_rule_unique(self):
        """Test calling multiple times when ``cardinality_rule`` is ``unique``."""
        # Setup
        data = pd.DataFrame({'my_column': np.arange(10)})
        generator = RegexGenerator(cardinality_rule='unique')

        # Run
        transformed_data = generator.fit_transform(data, 'my_column')
        first_reverse_transform = generator.reverse_transform(transformed_data.head(3))
        second_reverse_transform = generator.reverse_transform(transformed_data.head(5))

        # Assert
        expected_first_reverse_transform = pd.DataFrame({'my_column': ['AAAAA', 'AAAAB', 'AAAAC']})
        expected_second_reverse_transform = pd.DataFrame({
            'my_column': ['AAAAD', 'AAAAE', 'AAAAF', 'AAAAG', 'AAAAH']
        })
        pd.testing.assert_frame_equal(first_reverse_transform, expected_first_reverse_transform)
        pd.testing.assert_frame_equal(second_reverse_transform, expected_second_reverse_transform)

    def test_pickled(self, tmpdir):
        """Test that ensures that ``RegexGenerator`` can be pickled."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
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

    def test_with_many_possibilities(self):
        """Test the ``RegexGenerator`` with regex containing many possibilities."""
        # Setup
        data = pd.DataFrame({
            'id': [
                'a' * 50,
                'a' * 49 + 'b',
                'a' * 49 + 'c',
                'a' * 49 + 'd',
                'a' * 49 + 'e',
            ],
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
            'id': [
                'a' * 50,
                'a' * 49 + 'b',
                'a' * 49 + 'c',
                'a' * 49 + 'd',
                'a' * 49 + 'e',
            ],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transformed)

    def test_cardinality_rule_unique_not_enough_values_categorical(self):
        """Test with cardinality_rule='unique' but insufficient regex values."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
        })
        instance = RegexGenerator('id_[a-b]{1}', cardinality_rule='unique')

        # Run
        transformed = instance.fit_transform(data, 'id')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        expected = pd.DataFrame({'id': ['id_a', 'id_b', 'id_a(0)', 'id_b(0)', 'id_a(1)']})
        pd.testing.assert_frame_equal(reverse_transform, expected)

    def test_cardinality_rule_not_enough_values_numerical(self):
        """Test with cardinality_rule='unique' but insufficient regex values."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
        })
        instance = RegexGenerator('[2-3]{1}', cardinality_rule='unique')

        # Run
        transformed = instance.fit_transform(data, 'id')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        expected = pd.DataFrame({'id': ['2', '3', '4', '5', '6']}, dtype=object)
        pd.testing.assert_frame_equal(reverse_transform, expected)

    def test_cardinality_rule_match(self):
        """Test with cardinality_rule='match'."""
        # Setup
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
        instance = RegexGenerator('[1-3]{1}', cardinality_rule='match')

        # Run
        transformed = instance.fit_transform(data, 'id')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        expected = pd.DataFrame({'id': ['1', '2', '3', '4', '5']}, dtype=object)
        pd.testing.assert_frame_equal(reverse_transform, expected)

    def test_cardinality_rule_match_not_enough_values(self):
        """Test with cardinality_rule='match' but insufficient regex values."""
        # Setup
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
        instance = RegexGenerator('[1-3]{1}', cardinality_rule='match')

        # Run
        transformed = instance.fit_transform(data, 'id')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        expected = pd.DataFrame({'id': ['1', '2', '3', '4', '5']}, dtype=object)
        pd.testing.assert_frame_equal(reverse_transform, expected)

    def test_called_multiple_times_cardinality_rule_match(self):
        """Test calling multiple times when ``cardinality_rule`` is ``match``."""
        # Setup
        data = pd.DataFrame({'my_column': [1, 2, 3, 4, 5] * 3})
        generator = RegexGenerator(cardinality_rule='match')

        # Run
        transformed_data = generator.fit_transform(data, 'my_column')
        first_reverse_transform = generator.reverse_transform(transformed_data.head(3))
        second_reverse_transform = generator.reverse_transform(transformed_data.head(4))
        third_reverse_transform = generator.reverse_transform(transformed_data.head(5))
        fourth_reverse_transform = generator.reverse_transform(transformed_data.head(11))

        # Assert
        expected_first_reverse_transform = pd.DataFrame({'my_column': ['AAAAA', 'AAAAB', 'AAAAC']})
        expected_second_reverse_transform = pd.DataFrame({
            'my_column': ['AAAAA', 'AAAAB', 'AAAAC', 'AAAAD']
        })
        expected_third_reverse_transform = pd.DataFrame({
            'my_column': ['AAAAA', 'AAAAB', 'AAAAC', 'AAAAD', 'AAAAE']
        })
        expected_fourth_reverse_transform = pd.DataFrame({
            'my_column': [
                'AAAAA',
                'AAAAB',
                'AAAAC',
                'AAAAD',
                'AAAAE',
                'AAAAA',
                'AAAAB',
                'AAAAC',
                'AAAAD',
                'AAAAE',
                'AAAAA',
            ]
        })
        pd.testing.assert_frame_equal(first_reverse_transform, expected_first_reverse_transform)
        pd.testing.assert_frame_equal(second_reverse_transform, expected_second_reverse_transform)
        pd.testing.assert_frame_equal(third_reverse_transform, expected_third_reverse_transform)
        pd.testing.assert_frame_equal(fourth_reverse_transform, expected_fourth_reverse_transform)

    def test_cardinality_rule_match_empty_regex(self):
        """Test with cardinality_rule='match' but insufficient regex values."""
        # Setup
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
        instance_unique = RegexGenerator('', cardinality_rule='unique')
        instance_match = RegexGenerator('', cardinality_rule='match')
        instance_none = RegexGenerator('')

        # Run
        transformed_unique = instance_unique.fit_transform(data, 'id')
        transformed_match = instance_match.fit_transform(data, 'id')
        transformed_none = instance_none.fit_transform(data, 'id')
        reverse_transform_unique = instance_unique.reverse_transform(transformed_unique)
        reverse_transform_match = instance_match.reverse_transform(transformed_match)
        reverse_transform_none = instance_none.reverse_transform(transformed_none)

        # Assert
        expected = pd.DataFrame({'id': ['', '', '', '', '']})
        pd.testing.assert_frame_equal(reverse_transform_unique, expected)
        pd.testing.assert_frame_equal(reverse_transform_match, expected)
        pd.testing.assert_frame_equal(reverse_transform_none, expected)


class TestHyperTransformer:
    def test_end_to_end_scrambled(self):
        """Test the ``RegexGenerator`` in the ``HyperTransformer``.

        Check that when the ``generation_order`` is set to scrambled, the output data is
        scrambled.
        """
        # Setup
        customers = get_demo()
        customers['id'] = ['id_a', 'id_b', 'id_c', 'id_d', 'id_e']
        ht = HyperTransformer()
        ht.detect_initial_config(customers)
        ht.update_sdtypes({'id': 'id'})
        ht.update_transformers({
            'id': RegexGenerator(regex_format='id_[a-z]', generation_order='scrambled')
        })

        # Run
        ht.fit(customers)
        transformed = ht.transform(customers)
        reverse_transformed = ht.reverse_transform(transformed)

        # Assert
        expected_id = pd.Series(['id_b', 'id_c', 'id_a', 'id_d', 'id_e'], name='id')
        pd.testing.assert_series_equal(reverse_transformed['id'], expected_id)
