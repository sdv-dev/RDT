"""Functions for downloading datasets for quality testing."""

import io
import json
import urllib.request
from zipfile import ZipFile

import pandas as pd

DATA_URL = 'https://sdv-datasets.s3.amazonaws.com/{}.zip'


def download_dataset(dataset_name):
    """Download all tables for a dataset.

    Args:
        dataset_name (str):
            Name of dataset to download.

    Returns:
        Dict mapping table names to tuples of their DataFrames and metadata.
    """
    url = DATA_URL.format(dataset_name)
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())
    tables_dict = {}

    with ZipFile(bytes_io) as zf:
        with zf.open(f'{dataset_name}/metadata.json') as metadata_file:
            metadata = json.load(metadata_file)
        tables = metadata['tables']
        for table in tables:
            table_meta = tables[table]
            file_name = table_meta['path']
            file_path = f'{dataset_name}/{file_name}'
            with zf.open(file_path) as table_file:
                tables_dict[table] = (pd.read_csv(table_file), table_meta)

    return tables_dict


def download_single_table(dataset_name, table_name):
    """Download data for a single table dataset.

    Args:
        dataset_name (str):
            Name of dataset the table belongs to.
        table_name (str):
            Name of table to download.

    Returns:
        Tuple of DataFrame and metadata
    """
    tables_dict = download_dataset(dataset_name)
    return tables_dict[table_name]
