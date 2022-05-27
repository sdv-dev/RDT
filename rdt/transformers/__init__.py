"""Transformers module."""

import importlib
import inspect
import json
import sys
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from rdt.transformers.base import BaseTransformer
from rdt.transformers.boolean import BinaryEncoder
from rdt.transformers.categorical import (
    CustomLabelEncoder, FrequencyEncoder, LabelEncoder, OneHotEncoder)
from rdt.transformers.datetime import OptimizedTimestampEncoder, UnixTimestampEncoder
from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import ClusterBasedNormalizer, FloatFormatter, GaussianNormalizer
from rdt.transformers.pii.anonymizer import AnonymizedFaker
from rdt.transformers.text import RegexGenerator

__all__ = [
    'BaseTransformer',
    'BinaryEncoder',
    'ClusterBasedNormalizer',
    'CustomLabelEncoder',
    'FloatFormatter',
    'FrequencyEncoder',
    'GaussianNormalizer',
    'LabelEncoder',
    'NullTransformer',
    'OneHotEncoder',
    'OptimizedTimestampEncoder',
    'UnixTimestampEncoder',
    'RegexGenerator',
    'AnonymizedFaker',
    'get_transformer_name',
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


def get_transformer_name(transformer):
    """Return the fully qualified path of the transformer.

    Args:
        transformer:
            A transformer class.

    Raises:
        ValueError:
            Crashes when the transformer is not passed as a class.

    Returns:
        string:
            The path of the transformer.
    """
    if inspect.isclass(transformer):
        return transformer.__module__ + '.' + transformer.__name__

    raise ValueError(f'The transformer {transformer} must be passed as a class.')


TRANSFORMERS = {
    get_transformer_name(transformer): transformer
    for transformer in BaseTransformer.get_subclasses()
}


DEFAULT_TRANSFORMERS = {
    'numerical': FloatFormatter(missing_value_replacement='mean'),
    'categorical': FrequencyEncoder(),
    'boolean': BinaryEncoder(missing_value_replacement='mode'),
    'datetime': UnixTimestampEncoder(missing_value_replacement='mean'),
}


def get_transformer_class(transformer):
    """Return a ``transformer`` class from a ``str``.

    Args:
        transformer (str):
            Python path.

    Returns:
        BaseTransformer:
            BaseTransformer subclass class object.
    """
    if transformer in TRANSFORMERS:
        return TRANSFORMERS[transformer]

    package, name = transformer.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


def get_transformer_instance(transformer):
    """Load a new instance of a ``Transformer``.

    The ``transformer`` is expected to be the transformers path as a ``string``,
    a transformer instance or a transformer type.

    Args:
        transformer (str or BaseTransformer):
            String with the transformer path or instance of a BaseTransformer subclass.

    Returns:
        BaseTransformer:
            BaseTransformer subclass instance.
    """
    if isinstance(transformer, BaseTransformer):
        return deepcopy(transformer)

    if inspect.isclass(transformer) and issubclass(transformer, BaseTransformer):
        return transformer()

    return get_transformer_class(transformer)()


@lru_cache()
def get_transformers_by_type():
    """Build a ``dict`` mapping sdtypes to valid existing transformers for that sdtype.

    Returns:
        dict:
            Mapping of sdtypes to a list of existing transformers that take that
            sdtype as an input.
    """
    sdtype_transformers = defaultdict(list)
    transformer_classes = BaseTransformer.get_subclasses()
    for transformer in transformer_classes:
        input_sdtype = transformer.get_input_sdtype()
        sdtype_transformers[input_sdtype].append(transformer)

    return sdtype_transformers


@lru_cache()
def get_default_transformers():
    """Build a ``dict`` mapping sdtypes to a default transformer for that sdtype.

    Returns:
        dict:
            Mapping of sdtypes to a transformer.
    """
    transformers_by_type = get_transformers_by_type()
    defaults = deepcopy(DEFAULT_TRANSFORMERS)
    for (sdtype, transformers) in transformers_by_type.items():
        if sdtype not in defaults:
            defaults[sdtype] = transformers[0]()

    return defaults


@lru_cache()
def get_default_transformer(sdtype):
    """Get default transformer for a sdtype.

    Returns:
        Transformer:
            Default transformer for sdtype.
    """
    default_transformers = get_default_transformers()
    return default_transformers[sdtype]
