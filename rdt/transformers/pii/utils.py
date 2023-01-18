"""Utility functions for anonymization."""

import inspect

from faker import Faker


def is_faker_function(function_name):
    """Return whether or not the function name is a valid Faker function.

    Args:
        function_name (str):
            String representing a ``faker`` function.

    Returns:
        True if the ``function_name`` is know to ``Faker``, otherwise False.
    """
    try:
        getattr(Faker(), function_name)
    except AttributeError:
        return False

    return True


def get_provider_name(function_name):
    """Return the ``faker`` provider name for a given ``function_name``.

    Args:
        function_name (str):
            String representing a ``faker`` function.

    Returns:
        provider_name (str):
            String representing the provider name of the faker function.
    """
    function_name = getattr(Faker(), function_name)
    module = inspect.getmodule(function_name).__name__
    module = module.split('.')
    if len(module) == 2:
        return 'BaseProvider'

    return module[-1]
