from rdt.transformers.pii.utils import get_provider_name


def test_get_provider_name():
    """Test the ``get_provider_name`` method.

    Test that the function returns an expected provider name from the ``faker.Faker`` instance.
    If this is from the ``BaseProvider`` it should also return that name.
    """
    # Run
    email_provider = get_provider_name('email')
    lexify_provider = get_provider_name('lexify')

    # Assert
    assert email_provider == 'internet'
    assert lexify_provider == 'BaseProvider'
