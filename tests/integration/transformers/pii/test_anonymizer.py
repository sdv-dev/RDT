
import numpy as np
import pandas as pd

from rdt.transformers.pii import AnonymizedFaker


def test_anonymizedfaker():
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


def test_anonymizedfaker_custom_provider():
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


def test_anonymizedfaker_with_nans():
    """End to end test with the default settings of the ``AnonymizedFaker`` with ``nan`` values."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', np.nan, 'c', 'd', 'e']
    })

    instance = AnonymizedFaker(model_missing_values=True)
    transformed = instance.fit_transform(data, 'username')
    reverse_transform = instance.reverse_transform(transformed)

    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username.is_null': [0.0, 1.0, 0.0, 0.0, 0.0]
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    assert len(reverse_transform['username']) == 5
    assert reverse_transform['username'].isna().sum() == 1


def test_anonymizedfaker_custom_provider_with_nans():
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
        model_missing_values=True
    )
    transformed = instance.fit_transform(data, 'cc')
    reverse_transform = instance.reverse_transform(transformed)

    expected_transformed = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'username': ['a', 'b', 'c', 'd', 'e'],
        'cc.is_null': [0.0, 1.0, 0.0, 0.0, 0.0]
    })

    pd.testing.assert_frame_equal(transformed, expected_transformed)
    assert len(reverse_transform['cc']) == 5
    assert reverse_transform['cc'].isna().sum() == 1
