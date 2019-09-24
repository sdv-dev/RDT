# -*- coding: utf-8 -*-

"""Top-level package for RDT."""


__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.3'

import json
import os

import pandas as pd

from rdt.hyper_transformer import HyperTransformer


def _lookup(elements, field, value):
    for element in elements:
        if element[field] == value:
            return element

    raise ValueError('Invalid {}: {}'.format(field, value))


def load_data(metadata_path, table_name, column_name=None):
    """Load the metadata and data from the indicated table.

    If a column name is also given, restrict the data and metadata results
    to the indicated column.

    Args:
        metadata_path (str):
            Path to the metadata file.
        table_name (str):
            Name of the table.
        column_name (str):
            Name of the column. Optional.

    Returns:
        pandas.DataFrame, dict
            * Table or column loaded as a ``pandas.DataFrame``.
            * The table or column metadata.
    """
    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    table_metadata = _lookup(metadata['tables'], 'name', table_name)

    data_path = os.path.join(
        os.path.dirname(metadata_path),
        metadata['path'],
        table_metadata['path']
    )
    table_data = pd.read_csv(data_path)

    if column_name is None:
        return table_data, table_metadata

    else:
        column_metadata = _lookup(table_metadata['fields'], 'name', column_name)
        column_data = table_data[column_name].to_frame()

        return column_data, column_metadata


__all__ = [
    'HyperTransformer',
    'load_data'
]
