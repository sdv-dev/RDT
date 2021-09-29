from rdt.transformers import BooleanTransformer, load_transformer


def test_load_transformer_instance():
    transformer = BooleanTransformer(nan=None)

    returned = load_transformer(transformer)

    assert isinstance(returned, BooleanTransformer)
    assert returned.nan is None


def test_load_transformer_str():
    transformer = 'BooleanTransformer'

    returned = load_transformer(transformer)

    assert isinstance(returned, BooleanTransformer)


def test_load_transformer_class():
    transformer = BooleanTransformer

    returned = load_transformer(transformer)

    assert isinstance(returned, BooleanTransformer)
