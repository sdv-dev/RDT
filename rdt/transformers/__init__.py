"""Transformers module."""

import importlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from rdt.transformers.base import BaseTransformer
from rdt.transformers.boolean import BooleanTransformer
from rdt.transformers.categorical import CategoricalTransformer
from rdt.transformers.datetime import DatetimeTransformer
from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import NumericalTransformer

__all__ = [
    'BaseTransformer',
    'NullTransformer',
    'get_transformer_class',
    'get_transformer_instance',
    'get_transformers_by_type',
    'get_default_transformers',
    'get_default_transformer',
]


def _import_addons():
    """Import all the addon modules."""
    addons_path = Path(__file__).parent / 'addons'
    for addon_json_path in addons_path.glob('*/*.json'):
        with open(addon_json_path, 'r', encoding='utf-8') as addon_json_file:
            transformers = json.load(addon_json_file).get('transformers', [])
            for transformer in transformers:
                module = transformer.rsplit('.', 1)[0]
                if module not in sys.modules:
                    importlib.import_module(module)


_import_addons()

TRANSFORMERS = {
    transformer.__name__: transformer
    for transformer in BaseTransformer.get_subclasses()
}

globals().update(TRANSFORMERS)
__all__.extend(TRANSFORMERS.keys())

DEFAULT_TRANSFORMERS = {
    'numerical': NumericalTransformer,
    'integer': NumericalTransformer(dtype=np.int64),
    'float': NumericalTransformer(dtype=np.float64),
    'categorical': CategoricalTransformer(fuzzy=True),
    'boolean': BooleanTransformer,
    'datetime': DatetimeTransformer,
}


def get_transformer_class(transformer):
    """Return a ``transformer`` class from a ``str``.

    Args:
        transforemr (str):
            Python path or transformer's name.

    Returns:
        BaseTransformer:
            BaseTransformer subclass class object.
    """
    if len(transformer.split('.')) == 1:
        return TRANSFORMERS[transformer]

    package, name = transformer.rsplit('.', 1)
    return TRANSFORMERS.get(name, getattr(importlib.import_module(package), name))


def get_transformer_instance(transformer):
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


@lru_cache()
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
        input_type = transformer.get_input_type()
        data_type_transformers[input_type].append(transformer)

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
