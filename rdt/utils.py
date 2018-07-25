import json
import os.path as op

import pandas as pd


def get_table_dict(meta_file):
    """
    This function parses through a meta file and extracts the tables

    Returns dictionary mapping table name to tuple of (table, table_meta)
    """
    table_dict = {}

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    for table in meta['tables']:
        if table['use']:
            prefix = op.dirname(meta_file)
            relative_path = op.join(prefix, meta['path'], table['path'])
            data_table = pd.read_csv(relative_path)
            table_dict[table['name']] = (data_table, table)

    return table_dict


def get_transformers_dict(meta_file):
    """
    This function parses through a meta file and extracts the transformer info
    Returns dictionary mapping (table_name, col_name) => transformer
    """
    transformer_dict = {}
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    for table in meta['tables']:
        table_name = table['name']

        for field in table['fields']:
            col_name = field['name']
            if 'transformer' in field:
                transformer_dict[(table_name, col_name)] = field['transformer']

    return transformer_dict


def load_data_table(table_name, meta_file, meta):
    for table in meta['tables']:
        if table['name'] == table_name:
            prefix = op.dirname(meta_file)
            relative_path = op.join(prefix, meta['path'], table['path'])
            return pd.read_csv(relative_path), table


def get_col_info(table_name, col_name, meta_file):
    """
    This functions returns a tuple of a column and its
    corresponding meta info
    """

    data_table = None
    col_meta = None
    col = None

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    data_table, table = load_data_table(table_name, meta_file, meta)

    for field in table['fields']:
        if field['name'] == col_name:
            col_meta = field

    col = data_table[col_name]

    return (col, col_meta)
