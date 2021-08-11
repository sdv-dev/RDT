from rdt.transformers import (
    BooleanTransformer, DatetimeTransformer, NumericalTransformer, load_transformer,
    load_transformers)


def test_load_transformer_instance():
    transformer = BooleanTransformer()

    returned = load_transformer(transformer)

    assert returned is transformer


def test_load_transformer_str():
    transformer = {
        'class': 'BooleanTransformer',
    }

    returned = load_transformer(transformer)

    assert isinstance(returned, BooleanTransformer)


def test_load_transformer_class():
    transformer = {
        'class': BooleanTransformer,
    }

    returned = load_transformer(transformer)

    assert isinstance(returned, BooleanTransformer)


def test_load_transformer_kwargs():
    transformer = {
        'class': BooleanTransformer,
        'kwargs': {
            'nan': None
        }
    }

    returned = load_transformer(transformer)

    assert returned.nan is None


def test_load_transformers():
    transformers = {
        'bool': BooleanTransformer(),
        'int': {
            'class': 'NumericalTransformer',
            'kwargs': {
                'dtype': 'int'
            }
        },
        'datetime': {
            'class': DatetimeTransformer,
        }
    }

    returned = load_transformers(transformers)

    assert isinstance(returned, dict)
    assert set(returned.keys()) == {'bool', 'int', 'datetime'}
    assert isinstance(returned['bool'], BooleanTransformer)
    assert isinstance(returned['int'], NumericalTransformer)
    assert returned['int'].dtype == 'int'
    assert isinstance(returned['datetime'], DatetimeTransformer)
