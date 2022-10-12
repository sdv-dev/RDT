from rdt.transformers.pii.utils import get_provider_name, is_faker_function


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


def test_is_faker_function():
    """Test that the method returns True if the ``function_name`` is a valid faker function."""
    # Run
    result = is_faker_function('address')

    # Assert
    assert result is True


def test_is_faker_function_error():
    """Test that the method returns False if ``function_name`` is not a valid faker function."""
    # Run
    result = is_faker_function('blah')

    # Assert
    assert result is False
