import pytest

from rdt.transformers import (
    AnonymizedFaker, BinaryEncoder, FloatFormatter, RegexGenerator, UniformEncoder,
    UnixTimestampEncoder, get_default_transformers, get_transformer_class, get_transformer_name)
from rdt.transformers.addons.identity.identity import IdentityTransformer


def test_get_transformer_name():
    """Test the ``get_transformer_name`` method.

    Validate the method returns the class path when passed the class.

    Input:
        - a class.

    Output:
        - the path of the class.
    """
    # Setup
    transformer = BinaryEncoder

    # Run
    returned = get_transformer_name(transformer)

    # Assert
    assert returned == 'rdt.transformers.boolean.BinaryEncoder'


def test_get_transformer_name_incorrect_input():
    """Test the ``get_transformer_name`` method crashes.

    Validate the method raises a ``ValueError`` when passed a string.

    Input:
        - a string.

    Raises:
        - ``ValueError``, with the correct output message.
    """
    # Setup
    transformer = 'rdt.transformers.boolean.BinaryEncoder'

    # Run / Assert
    error_msg = 'The transformer rdt.transformers.boolean.BinaryEncoder must be passed as a class.'
    with pytest.raises(ValueError, match=error_msg):
        get_transformer_name(transformer)


def test_get_transformer_class_transformer_path():
    """Test the ``get_transformer_class`` method.

    Validate the method returns the correct class when passed the class path.

    Input:
        - a string describing the transformer path.

    Output:
        - the class corresponding to the transformer path.
    """
    # Setup
    transformer_path = 'rdt.transformers.boolean.BinaryEncoder'

    # Run
    returned = get_transformer_class(transformer_path)

    # Assert
    assert returned == BinaryEncoder


def test_get_transformer_class_partial_path():
    """Test with non fully specified path."""
    # Run
    returned = get_transformer_class('rdt.transformers.BinaryEncoder')

    # Assert
    assert returned == BinaryEncoder


def test_get_transformer_class_transformer_path_addon():
    """Test the ``get_transformer_class`` method.

    Validate the method returns the correct class when passed an addon path.

    Input:
        - a string describing the transformer path.

    Output:
        - the class corresponding to the transformer path.
    """
    # Setup
    transformer_path = 'rdt.transformers.addons.identity.identity.IdentityTransformer'

    # Run
    returned = get_transformer_class(transformer_path)

    # Assert
    assert returned == IdentityTransformer


def test_get_default_transformers():
    """Test the ``get_default_transformers`` method.

    Check that the right default transformer is returned for each type.
    """
    # Run
    default_transformer_dict = get_default_transformers()

    # Assert
    expected_dict = {
        'numerical': FloatFormatter,
        'categorical': UniformEncoder,
        'boolean': UniformEncoder,
        'datetime': UnixTimestampEncoder,
        'text': RegexGenerator,
        'pii': AnonymizedFaker,
    }

    for sdtype, transformer in expected_dict.items():
        assert isinstance(default_transformer_dict[sdtype], transformer)
