import re

import numpy as np
import pandas as pd
import pytest

from rdt.errors import TransformerProcessingError
from rdt.transformers.pii import AnonymizedFaker, PseudoAnonymizedFaker


class TestAnonymizedFaker:
    def test_default_settings(self):
        """End to end test with the default settings of the ``AnonymizedFaker``."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e']
        })

        instance = AnonymizedFaker()
        transformed = instance.fit_transform(data, 'username')

        reverse_transform = instance.reverse_transform(transformed)
        expected_transformed = pd.DataFrame({
            'id': [1, 2, 3, 4, 5]
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        assert len(reverse_transform['username']) == 5

    def test_get_supported_sdtypes(self):
        """Test that the correct supported sdtypes are returned."""
        # Run
        supported_sdtypes = AnonymizedFaker.get_supported_sdtypes()

        # Assert
        assert sorted(supported_sdtypes) == sorted(['pii', 'text'])

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
                '213133122335401'
            ]
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
            'username': ['a', np.nan, 'c', 'd', 'e']
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
            'username': ['a', np.nan, 'c', 'd', 'e']
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
                '213133122335401'
            ]
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

    def test_enforce_uniqueness(self):
        """Test that ``AnonymizedFaker`` works with uniqueness.

        Also ensure that when we call ``reset_randomization`` the generator will be able to
        create values again.
        """
        data = pd.DataFrame({
            'job': np.arange(500)
        })

        instance = AnonymizedFaker('job', 'job', enforce_uniqueness=True)
        transformed = instance.fit_transform(data, 'job')
        reverse_transform = instance.reverse_transform(transformed)

        # Assert
        assert len(reverse_transform['job'].unique()) == 500

        error_msg = re.escape(
            'The Faker function you specified is not able to generate 500 unique '
            'values. Please use a different Faker function for column '
            "('job')."
        )
        with pytest.raises(TransformerProcessingError, match=error_msg):
            instance.reverse_transform(transformed)

        instance.reset_randomization()
        instance.reverse_transform(transformed)


class TestPsuedoAnonymizedFaker:
    def test_default_settings(self):
        """End to end test with the default settings of the ``PseudoAnonymizedFaker``."""
        data = pd.DataFrame({
            'animals': ['cat', 'dog', 'parrot', 'monkey']
        })

        instance = PseudoAnonymizedFaker()

        transformed = instance.fit_transform(data, 'animals')
        reverse_transformed = instance.reverse_transform(transformed)

        assert transformed.columns == ['animals']
        pd.testing.assert_series_equal(
            reverse_transformed['animals'].map(instance._reverse_mapping_dict),
            data['animals']
        )
        unique_animals = set(reverse_transformed['animals'])
        assert unique_animals.intersection(set(instance._mapping_dict)) == set()
        assert len(reverse_transformed) == len(transformed) == 4

    def test_with_nans(self):
        """Test with the default settings of the ``PseudoAnonymizedFaker`` and ``nans``."""
        data = pd.DataFrame({
            'animals': ['cat', 'dog', np.nan, 'monkey']
        })

        instance = PseudoAnonymizedFaker()

        transformed = instance.fit_transform(data, 'animals')
        reverse_transformed = instance.reverse_transform(transformed)

        assert transformed.columns == ['animals']
        pd.testing.assert_series_equal(
            reverse_transformed['animals'].map(instance._reverse_mapping_dict),
            data['animals']
        )
        unique_animals = set(reverse_transformed['animals'])
        assert unique_animals.intersection(set(instance._mapping_dict)) == set()
        assert len(reverse_transformed) == len(transformed) == 4

    def test_with_custom_provider(self):
        """End to end test with custom settings of the ``PseudoAnonymizedFaker``."""
        data = pd.DataFrame({
            'animals': ['cat', 'dog', np.nan, 'monkey']
        })

        instance = PseudoAnonymizedFaker('credit_card', 'credit_card_number')

        transformed = instance.fit_transform(data, 'animals')
        reverse_transformed = instance.reverse_transform(transformed)

        assert transformed.columns == ['animals']
        pd.testing.assert_series_equal(
            reverse_transformed['animals'].map(instance._reverse_mapping_dict),
            data['animals']
        )
        unique_animals = set(reverse_transformed['animals'])
        assert unique_animals.intersection(set(instance._mapping_dict)) == set()
        assert len(reverse_transformed) == len(transformed) == 4
