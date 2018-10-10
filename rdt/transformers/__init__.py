import json
import os

import pandas as pd

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.CatTransformer import CatTransformer
from rdt.transformers.DTTransformer import DTTransformer
from rdt.transformers.NullTransformer import NullTransformer
from rdt.transformers.NumberTransformer import NumberTransformer


def load_data_table(table_name, meta_file, meta):
    """Return the contents and metadata of a given table.

    Args:
        table_name(str): Name of the table.
        meta_file(str): Path to the meta.json file.
        meta(dict): Contents of meta.json.

    Returns:
        tuple(pandas.DataFrame, dict)

    """
    for table in meta['tables']:
        if table['name'] == table_name:
            prefix = os.path.dirname(meta_file)
            relative_path = os.path.join(prefix, meta['path'], table['path'])
            return pd.read_csv(relative_path), table


def get_col_info(table_name, col_name, meta_file):
    """Return the content and metadata of a fiven column.

    Args:
        table_name(str): Name of the table.
        col_name(str): Name of the column.
        meta_file(str): Path to the meta.json file.

    Returns:
        tuple(pandas.Series, dict)
    """

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    data_table, table = load_data_table(table_name, meta_file, meta)

    for field in table['fields']:
        if field['name'] == col_name:
            col_meta = field

    col = data_table[col_name]

    return (col, col_meta)


__all__ = [
    'BaseTransformer',
    'CatTransformer',
    'DTTransformer',
    'NullTransformer',
    'NumberTransformer',
    'get_col_info'
]
