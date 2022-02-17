from rdt.transformers import BinaryEncoder, get_transformer_class, get_transformer_instance


def test_get_transformer_class_transformer_path():
    """Test the ``get_transformer_class`` method.

    Validate the method returns the correct class when passed the class path.

    Input:
        - a string describing the transformer path.

    Output:
        - the class corresponding to the transformer path.
    """
    # Setup
    transformer_path = 'rdt.transformers.BinaryEncoder'

    # Run
    returned = get_transformer_class(transformer_path)

    # Assert
    assert returned == BinaryEncoder


def test_get_transformer_instance_instance():
    transformer = BinaryEncoder(missing_value_replacement=None)

    returned = get_transformer_instance(transformer)

    assert isinstance(returned, BinaryEncoder)
    assert returned.missing_value_replacement is None


def test_get_transformer_instance_str():
    transformer = 'rdt.transformers.BinaryEncoder'

    returned = get_transformer_instance(transformer)

    assert isinstance(returned, BinaryEncoder)


def test_get_transformer_instance_class():
    transformer = BinaryEncoder

    returned = get_transformer_instance(transformer)

    assert isinstance(returned, BinaryEncoder)
