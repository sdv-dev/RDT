"""Transformers module."""

from rdt.transformers.base import BaseTransformer
from rdt.transformers.boolean import BooleanTransformer
from rdt.transformers.categorical import (
    CategoricalTransformer, LabelEncodingTransformer, OneHotEncodingTransformer)
from rdt.transformers.datetime import DatetimeTransformer
from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import GaussianCopulaTransformer, NumericalTransformer

__all__ = [
    'BaseTransformer',
    'BooleanTransformer',
    'CategoricalTransformer',
    'DatetimeTransformer',
    'GaussianCopulaTransformer',
    'NumericalTransformer',
    'NullTransformer',
    'OneHotEncodingTransformer',
    'LabelEncodingTransformer',
]


TRANSFORMERS = {
    transformer.__name__: transformer
    for transformer in BaseTransformer.__subclasses__()
}


def load_transformer(transformer):
    """Load a new instance of a ``Transformer``.

    The ``transformer`` is expected to be a ``dict`` containing  the transformer ``class``
    and its ``kwargs``. The ``class`` entry can either be the actual ``class`` or its name.
    For convenience, if an instance of a ``BaseTransformer`` is passed, it will be
    returned unmodified.

    Args:
        transformer (dict or BaseTransformer):
            ``dict`` with the transformer specification or instance of a BaseTransformer
            subclass.

    Returns:
        BaseTransformer:
            BaseTransformer subclass instance.
    """
    if isinstance(transformer, BaseTransformer):
        return transformer

    transformer_class = transformer['class']
    if isinstance(transformer_class, str):
        transformer_class = TRANSFORMERS[transformer_class]

    transformer_kwargs = transformer.get('kwargs')
    if transformer_kwargs is None:
        transformer_kwargs = dict()

    return transformer_class(**transformer_kwargs)


def load_transformers(transformers):
    """Load a dict of transfomers from a dict specification.

    >>> nt = NumericalTransformer(dtype=float)
    >>> transformers = {
    ...     'a': nt,
    ...     'b': {
    ...         'class': 'NumericalTransformer',
    ...         'kwargs': {
    ...             'dtype': int
    ...         }
    ...     }
    ... }
    >>> load_transformers(transformers)
    """
    return {
        name: load_transformer(transformer)
        for name, transformer in transformers.items()
    }
