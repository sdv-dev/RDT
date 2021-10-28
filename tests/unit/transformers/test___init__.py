from rdt.transformers import BooleanTransformer, get_transformer_class, get_transformer_instance


def test_get_transformer_class_transformer_name():
    """Test the ``get_transformer_class`` method.

    Validate the method returns the correct class when passed the class name.

    Input:
        - a string describing the transformer name.

    Output:
        - the class corresponding to the transformer name.
    """
    # Setup
    transformer_name = 'BooleanTransformer'

    # Run
    returned = get_transformer_class(transformer_name)

    # Assert
    assert returned == BooleanTransformer


def test_get_transformer_class_transformer_path():
    """Test the ``get_transformer_class`` method.

    Validate the method returns the correct class when passed the class path.

    Input:
        - a string describing the transformer path.

    Output:
        - the class corresponding to the transformer path.
    """
    # Setup
    transformer_path = 'rdt.transformers.BooleanTransformer'

    # Run
    returned = get_transformer_class(transformer_path)

    # Assert
    assert returned == BooleanTransformer


def test_get_transformer_instance_instance():
    transformer = BooleanTransformer(nan=None)

    returned = get_transformer_instance(transformer)

    assert isinstance(returned, BooleanTransformer)
    assert returned.nan is None


def test_get_transformer_instance_str():
    transformer = 'BooleanTransformer'

    returned = get_transformer_instance(transformer)

    assert isinstance(returned, BooleanTransformer)


def test_get_transformer_instance_class():
    transformer = BooleanTransformer

    returned = get_transformer_instance(transformer)

    assert isinstance(returned, BooleanTransformer)
