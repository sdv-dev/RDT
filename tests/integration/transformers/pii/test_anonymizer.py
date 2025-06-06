import re

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.pii import AnonymizedFaker, PseudoAnonymizedFaker


class TestAnonymizedFaker:
    def test_default_settings(self):
        """End to end test with the default settings of the ``AnonymizedFaker``."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
        })

        instance = AnonymizedFaker()
        transformed = instance.fit_transform(data, 'username')

        reverse_transform = instance.reverse_transform(transformed)
        expected_transformed = pd.DataFrame({'id': [1, 2, 3, 4, 5]})

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['username']) == 5

    def test_default_settings_with_locales(self):
        """End to end test with the default settings and locales of the ``AnonymizedFaker``."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
        })

        instance = AnonymizedFaker(locales=['en_US', 'en_CA', 'es_ES'])
        transformed = instance.fit_transform(data, 'username')

        reverse_transform = instance.reverse_transform(transformed)
        expected_transformed = pd.DataFrame({'id': [1, 2, 3, 4, 5]})

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['username']) == 5

    def test_get_supported_sdtypes(self):
        """Test that the correct supported sdtypes are returned."""
        # Run
        supported_sdtypes = AnonymizedFaker.get_supported_sdtypes()

        # Assert
        assert sorted(supported_sdtypes) == sorted(['pii', 'text', 'id'])

    def test_custom_provider(self):
        """End to end test with a custom provider and function for the ``AnonymizedFaker``."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
            'cc': [
                '2276346007210438',
                '4149498289355',
                '213144860944676',
                '4514775286178',
                '213133122335401',
            ],
        })

        instance = AnonymizedFaker('credit_card', 'credit_card_number')
        transformed = instance.fit_transform(data, 'cc')
        reverse_transform = instance.reverse_transform(transformed)

        expected_transformed = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['cc']) == 5

    def test_with_nans(self):
        """Test with the default settings of the ``AnonymizedFaker`` with ``nan`` values."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', np.nan, 'c', 'd', 'e'],
        })

        instance = AnonymizedFaker()
        transformed = instance.fit_transform(data, 'username')
        reverse_transform = instance.reverse_transform(transformed)

        expected_transformed = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['username']) == 5
        assert reverse_transform['username'].isna().sum() == 1

    def test_with_nans_missing_value_generation_none(self):
        """End to end test settings missing_value_generation=None."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', np.nan, 'c', 'd', 'e'],
        })

        instance = AnonymizedFaker(missing_value_generation=None)
        transformed = instance.fit_transform(data, 'username')
        reverse_transform = instance.reverse_transform(transformed)

        expected_transformed = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['username']) == 5
        assert reverse_transform['username'].isna().sum() == 0

    def test_custom_provider_with_nans(self):
        """End to end test with a custom provider for the ``AnonymizedFaker`` with `` nans``."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
            'cc': [
                '2276346007210438',
                np.nan,
                '213144860944676',
                '4514775286178',
                '213133122335401',
            ],
        })

        instance = AnonymizedFaker(
            'credit_card',
            'credit_card_number',
        )
        transformed = instance.fit_transform(data, 'cc')
        reverse_transform = instance.reverse_transform(transformed)

        expected_transformed = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e'],
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['cc']) == 5
        assert reverse_transform['cc'].isna().sum() == 1

    def test_cardinality_rule(self):
        """Test that ``AnonymizedFaker`` works with uniqueness.

        Also ensure that when we call ``reset_randomization`` the generator will be able to
        create values again.
        """
        data = pd.DataFrame({'job': np.arange(500)})

        instance = AnonymizedFaker('job', 'job', cardinality_rule='unique')
        transformed = instance.fit_transform(data, 'job')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['job'].unique()) == 500

        warning_msg = re.escape(
            "Unable to generate enough unique values for column 'job' in "
            'a human-readable format. Additional values may be created randomly.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            instance.reverse_transform(transformed)

        instance.reset_randomization()
        instance.reverse_transform(transformed)

    def test_cardinality_rule_match(self):
        """Test it works with the cardinality rule 'match'."""
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3, 1, 2]})
        instance = AnonymizedFaker(cardinality_rule='match')

        # Run
        transformed = instance.fit_transform(data, 'col')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['col'].unique()) == 3

    def test_cardinality_rule_match_nans(self):
        """Test it works with the cardinality rule 'match' with nans."""
        # Setup
        data = pd.DataFrame({'col': [1, 2, 3, 1, 2, None, np.nan, np.nan, 2]})
        instance = AnonymizedFaker(cardinality_rule='match')

        # Run
        transformed = instance.fit_transform(data, 'col')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['col'].unique()) == 4
        assert reverse_transform['col'].isna().sum() == 3

    def test_cardinality_rule_match_not_enough_unique_values(self):
        """Test it works with the cardinality rule 'match' and too few values to transform."""
        # Setup
        data_fit = pd.DataFrame({'col': [1, 2, 3, 1, 2, None, np.nan, np.nan, 2]})
        data_transform = pd.DataFrame({'col': [1, 1, 1]})
        instance = AnonymizedFaker(cardinality_rule='match')

        # Run
        transformed = instance.fit(data_fit, 'col')
        transformed = instance.transform(data_transform)
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['col'].unique()) == 3
        assert reverse_transform['col'].isna().sum() == 1

    def test_cardinality_rule_match_too_many_unique(self):
        """Test it works with the cardinality rule 'match' and more unique values than samples."""
        # Setup
        data_fit = pd.DataFrame({'col': [1, 2, 3, 4, 5, 6]})
        data_transform = pd.DataFrame({'col': [1, 1, np.nan, 3, 1]})
        instance = AnonymizedFaker(cardinality_rule='match')

        # Run
        transformed = instance.fit(data_fit, 'col')
        transformed = instance.transform(data_transform)
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['col'].unique()) == 5
        assert reverse_transform['col'].isna().sum() == 0

    def test_cardinality_rule_match_too_many_nans(self):
        """Test it works with the cardinality rule 'match' and more nans than possible to fit."""
        # Setup
        data_fit = pd.DataFrame({'col': [1, 2, 3, np.nan, np.nan, np.nan]})
        data_transform = pd.DataFrame({'col': [1, 1, 1, 1]})
        instance = AnonymizedFaker(cardinality_rule='match')

        # Run
        transformed = instance.fit(data_fit, 'col')
        transformed = instance.transform(data_transform)
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['col'].unique()) == 3
        assert reverse_transform['col'].isna().sum() == 2

    def test_enforce_uniqueness_backwards_compatability(self):
        """Test that ``AnonymizedFaker`` is backwards compatible with ``enforce_uniqueness``.

        Checks that transformers without the ``cardinality_rule`` attribute still function as
        expected (can happen when previous transformer version is loaded from a pkl file).
        """
        # Setup
        data = pd.DataFrame({'job': np.arange(500)})

        instance = AnonymizedFaker('job', 'job', cardinality_rule='match')
        instance.enforce_uniqueness = True

        transformed = instance.fit_transform(data, 'job')
        delattr(instance, 'cardinality_rule')

        # Run
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['job'].unique()) == 500

        warning_msg = re.escape(
            "Unable to generate enough unique values for column 'job' in "
            'a human-readable format. Additional values may be created randomly.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            instance.reverse_transform(transformed)

        instance.reset_randomization()
        instance.reverse_transform(transformed)

    def test__reverse_transform_from_manually_set_parameters(self):
        """Test the ``reverse_transform`` after manually setting parameters."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
        })
        transformer = AnonymizedFaker(cardinality_rule='match')
        column_name = 'id'
        freq = 0.6
        cardinality = 5

        # Run
        transformer._set_fitted_parameters(
            column_name=column_name, cardinality=cardinality, nan_frequency=freq
        )
        output = transformer.reverse_transform(data)
        missing_values = output.isna().sum().sum()

        # Assert
        assert missing_values / output.size == freq

    def test_anonymized_faker_produces_only_n_values_for_each_reverse_transform_cardinality_match(
        self,
    ):
        """Test `AnonymizedFaker` when `cardinality_rule` is set to `match`.

        Ensure that the AnonymizedFaker transformer with `cardinality_rule='match'`
        maintains the correct number of unique values across multiple `reverse_transform` calls.
        """
        # Setup
        data = pd.DataFrame(data={'name': ['Amy'] * 10 + ['Bob'] * 20 + ['Carla'] * 50})
        transformer = AnonymizedFaker(
            provider_name='person', function_name='name', cardinality_rule='match'
        )

        # Run
        transformed_data = transformer.fit_transform(data, 'name')
        first_reverse_transformed = transformer.reverse_transform(transformed_data)

        transformed_again = transformer.transform(first_reverse_transformed)
        second_reverse_transformed = transformer.reverse_transform(transformed_again)

        # Assert
        assert set(first_reverse_transformed['name']) == set(second_reverse_transformed['name'])

    def test_cardinality_rule_scale(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 50 + ['B'] * 100})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed = instance.fit_transform(data, 'col')
        out = instance.reverse_transform(transformed)

        # Assert
        assert set(out['col']) == {'KAab', 'qOSU'}

        value_counts = out['col'].value_counts()
        assert value_counts['KAab'] == 50
        assert value_counts['qOSU'] == 100

    def test_cardinality_rule_scale_nans(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': [np.nan] * 50 + ['B'] * 100})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed = instance.fit_transform(data, 'col')
        out = instance.reverse_transform(transformed)

        # Assert
        value_counts = out['col'].value_counts()
        assert value_counts['MGWz'] == 100
        assert out['col'].isna().sum() == 50

    def test_cardinality_rule_scale_one_value(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 50})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed = instance.fit_transform(data, 'col')
        out = instance.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(out, pd.DataFrame({'col': ['qOSU'] * 50}))

    def test_cardinality_rule_scale_one_value_many_transform(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': ['A'] * 50})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        instance.fit_transform(data, 'col')
        out = instance.reverse_transform(pd.DataFrame(index=range(200)))

        # Assert
        value_counts = out['col'].value_counts()
        assert value_counts['qOSU'] == 50
        assert value_counts['JEWW'] == 50
        assert value_counts['KAab'] == 50
        assert value_counts['CPmg'] == 50

    def test_cardinality_rule_scale_empty_data(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        data = pd.DataFrame({'col': []})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed = instance.fit_transform(data, 'col')
        out = instance.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(out, data, check_dtype=False)

    def test_cardinality_rule_scale_proportions(self):
        """Test when cardinality rule is 'scale'."""
        # Setup
        once = list(range(1000))
        twice = [i // 2 for i in range(2000, 3000)]
        thrice = [i // 3 for i in range(4500, 5500)]
        data = pd.DataFrame({'col': once + twice + thrice})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed = instance.fit_transform(data, 'col')
        out = instance.reverse_transform(transformed)

        # Assert
        value_counts = out['col'].value_counts()
        one_count = (value_counts == 1).sum()
        two_count = (value_counts == 2).sum()
        three_count = (value_counts == 3).sum()
        more_count = (value_counts > 3).sum()

        assert 900 < one_count < 1100
        assert 400 < two_count < 600
        assert 233 < three_count < 433
        assert len(out) == 3000
        assert more_count == 0

    def assert_proportions(self, out, samples):
        value_counts = out['col'].value_counts()
        one_count = (value_counts == 1).sum()
        two_count = (value_counts == 2).sum()
        three_count = (value_counts == 3).sum()
        more_count = (value_counts > 3).sum()

        assert np.isclose(one_count, two_count * 2, atol=samples * 0.2)
        assert np.isclose(one_count, three_count * 3, atol=samples * 0.2)
        assert len(out) == samples
        assert more_count <= 1

    def test_cardinality_rule_scale_called_multiple_times(self):
        """Test calling multiple times when ``cardinality_rule`` is ``scale``."""
        # Setup
        once = list(range(1000))
        twice = [i // 2 for i in range(2000, 3000)]
        thrice = [i // 3 for i in range(4500, 5500)]
        data = pd.DataFrame({'col': once + twice + thrice})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed_data = instance.fit_transform(data, 'col')
        first_reverse_transform = instance.reverse_transform(transformed_data.head(500))
        second_reverse_transform = instance.reverse_transform(transformed_data.head(1000))
        third_reverse_transform = instance.reverse_transform(transformed_data.head(2000))
        fourth_reverse_transform = instance.reverse_transform(transformed_data.head(1111))

        # Assert
        self.assert_proportions(first_reverse_transform, 500)
        self.assert_proportions(second_reverse_transform, 1000)
        self.assert_proportions(third_reverse_transform, 2000)
        self.assert_proportions(fourth_reverse_transform, 1111)
        self.assert_proportions(
            pd.concat([
                first_reverse_transform,
                second_reverse_transform,
                third_reverse_transform,
                fourth_reverse_transform,
            ]),
            4611,
        )

        first_set = set(first_reverse_transform['col'])
        second_set = set(second_reverse_transform['col'])
        third_set = set(third_reverse_transform['col'])
        fourth_set = set(fourth_reverse_transform['col'])

        assert len(first_set.intersection(second_set)) <= 1
        assert len(first_set.intersection(third_set)) <= 1
        assert len(first_set.intersection(fourth_set)) <= 1
        assert len(second_set.intersection(third_set)) <= 1
        assert len(second_set.intersection(fourth_set)) <= 1
        assert len(third_set.intersection(fourth_set)) <= 1

    def test_cardinality_rule_scale_called_multiple_times_remaining_samples(self):
        """Test calling multiple times when ``cardinality_rule`` is ``scale``."""
        # Setup
        hundred = [i // 100 for i in range(1000)]
        two_hundred = [i // 200 for i in range(2000, 3000)]
        data = pd.DataFrame({'col': hundred + two_hundred})
        instance = AnonymizedFaker(cardinality_rule='scale')

        # Run
        transformed_data = instance.fit_transform(data, 'col')
        first_out = instance.reverse_transform(transformed_data.head(250))
        remaining_value = instance._remaining_samples['value']
        remaining_samples = instance._remaining_samples['repetitions']
        second_out = instance.reverse_transform(transformed_data)

        # Assert
        assert len(first_out) == 250
        assert len(first_out[first_out['col'] == remaining_value]) == 50
        assert len(second_out['col']) == 2_000
        assert len(second_out[second_out['col'] == remaining_value]) == remaining_samples


class TestPsuedoAnonymizedFaker:
    def test_default_settings(self):
        """End to end test with the default settings of the ``PseudoAnonymizedFaker``."""
        data = pd.DataFrame({'animals': ['cat', 'dog', 'parrot', 'monkey']})

        instance = PseudoAnonymizedFaker()

        transformed = instance.fit_transform(data, 'animals')
        reverse_transformed = instance.reverse_transform(transformed)

        assert transformed.columns == ['animals']
        pd.testing.assert_series_equal(
            reverse_transformed['animals'].map(instance._reverse_mapping_dict),
            data['animals'],
        )
        unique_animals = set(reverse_transformed['animals'])
        assert unique_animals.intersection(set(instance._mapping_dict)) == set()
        assert len(reverse_transformed) == len(transformed) == 4

    def test_with_nans(self):
        """Test with the default settings of the ``PseudoAnonymizedFaker`` and ``nans``."""
        data = pd.DataFrame({'animals': ['cat', 'dog', np.nan, 'monkey']})

        instance = PseudoAnonymizedFaker()

        transformed = instance.fit_transform(data, 'animals')
        reverse_transformed = instance.reverse_transform(transformed)

        assert transformed.columns == ['animals']
        pd.testing.assert_series_equal(
            reverse_transformed['animals'].map(instance._reverse_mapping_dict),
            data['animals'],
        )
        unique_animals = set(reverse_transformed['animals'])
        assert unique_animals.intersection(set(instance._mapping_dict)) == set()
        assert len(reverse_transformed) == len(transformed) == 4

    def test_with_custom_provider(self):
        """End to end test with custom settings of the ``PseudoAnonymizedFaker``."""
        data = pd.DataFrame({'animals': ['cat', 'dog', np.nan, 'monkey']})

        instance = PseudoAnonymizedFaker('credit_card', 'credit_card_number')

        transformed = instance.fit_transform(data, 'animals')
        reverse_transformed = instance.reverse_transform(transformed)

        assert transformed.columns == ['animals']
        pd.testing.assert_series_equal(
            reverse_transformed['animals'].map(instance._reverse_mapping_dict),
            data['animals'],
        )
        unique_animals = set(reverse_transformed['animals'])
        assert unique_animals.intersection(set(instance._mapping_dict)) == set()
        assert len(reverse_transformed) == len(transformed) == 4
