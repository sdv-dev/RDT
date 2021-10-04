"""Transformers module."""

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

import numpy as np

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
    for transformer in BaseTransformer.get_subclasses()
}
DEFAULT_TRANSFORMERS = {
    'numerical': NumericalTransformer,
    'integer': NumericalTransformer(dtype=np.int64),
    'float': NumericalTransformer(dtype=np.float64),
    'categorical': CategoricalTransformer(fuzzy=True),
    'boolean': BooleanTransformer,
    'datetime': DatetimeTransformer,
}


def load_transformer(transformer):
    """Load a new instance of a ``Transformer``.

    The ``transformer`` is expected to be a ``string`` containing  the transformer ``class``
    name, a transformer instance or a transformer type.

    Args:
        transformer (dict or BaseTransformer):
            ``dict`` with the transformer specification or instance of a BaseTransformer
            subclass.

    Returns:
        BaseTransformer:
            BaseTransformer subclass instance.
    """
    if isinstance(transformer, BaseTransformer):
        return deepcopy(transformer)

    if isinstance(transformer, str):
        transformer = TRANSFORMERS[transformer]

    return transformer()


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


def get_transformers_by_type():
    """Build a ``dict`` mapping data types to valid existing transformers for that type.

    Returns:
        dict:
            Mapping of data types to a list of existing transformers that take that
            type as an input.
    """
    data_type_transformers = defaultdict(list)
    transformer_classes = BaseTransformer.get_subclasses()
    for transformer in transformer_classes:
        try:
            input_type = transformer.get_input_type()
            data_type_transformers[input_type].append(transformer)
        except AttributeError:
            pass

    return data_type_transformers


@lru_cache()
def get_default_transformers():
    """Build a ``dict`` mapping data types to a default transformer for that type.

    Returns:
        dict:
            Mapping of data types to a transformer.
    """
    transformers_by_type = get_transformers_by_type()
    defaults = deepcopy(DEFAULT_TRANSFORMERS)
    for (data_type, transformers) in transformers_by_type.items():
        if data_type not in defaults:
            defaults[data_type] = transformers[0]

    return defaults


@lru_cache()
def get_default_transformer(data_type):
    """Get default transformer for a data type.

    Returns:
        Transformer:
            Default transformer for data type.
    """
    default_transformers = get_default_transformers()
    return default_transformers[data_type]
