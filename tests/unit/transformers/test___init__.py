from rdt.transformers import BooleanTransformer, get_transformer_instance


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
