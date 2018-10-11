import json
import os

import pandas as pd

from rdt import transformers

TRANSFORMERS = {
    'datetime': 'DTTransformer',
    'categorical': 'CatTransformer',
    'number': 'NumberTransformer'
}


class HyperTransformer(object):
    """Multitable transformer.

    Arguments:
        metadata(str or dict): Path to the meta.json file or its parsed contents.
        dir_name(str): Path to the root directory of meta.json.

    The main propouse of the HyperTransformer class is to easily manage transformations
    for a whole dataset.
    """

    def __init__(self, metadata, dir_name=None):

        self.transformers = {}  # key=(table_name, col_name) val=transformer

        if isinstance(metadata, str):
            dir_name = os.path.dirname(metadata)
            with open(metadata, 'r') as f:
                metadata = json.load(f)

        elif not isinstance(metadata, dict):
            raise ValueError('Incorrect type for metadata: It can only be either dict or str.')

        elif dir_name is None:
            raise ValueError('dir_name is required when metadata is a dict.')

        self.table_dict = self._get_tables(metadata, dir_name)
        self.transformer_dict = self._get_transformers(metadata)

    @staticmethod
    def _get_tables(meta, base_dir):
        """Load the contents of meta_file and the corresponding data.

        Args:
            meta(dict): Metadata for the dataset.
            base_dir(str): Root folder of the dataset files.

        Returns:
            dict: Mapping str -> tuple(pandas.DataFrame, dict)
        """
        table_dict = {}

        for table in meta['tables']:
            if table['use']:
                relative_path = os.path.join(base_dir, meta['path'], table['path'])
                data_table = pd.read_csv(relative_path)
                table_dict[table['name']] = (data_table, table)

        return table_dict

    @staticmethod
    def _get_transformers(meta):
        """Load the contents of meta_file and extract information about the transformers.

        Args:
            meta(dict): Metadata for the dataset.

        Returns:
            dict: tuple(str, str) -> Transformer.
        """
        transformer_dict = {}

        for table in meta['tables']:
            table_name = table['name']

            for field in table['fields']:
                transformer_type = field.get('type')
                if transformer_type:
                    col_name = field['name']
                    transformer_dict[(table_name, col_name)] = transformer_type

        return transformer_dict

    def get_class(self, class_name):
        """Get class object of transformer from its class name.

        Args:
            class_name(str):    Name of the transform.

        Returns:
            BaseTransformer
        """
        return getattr(transformers, class_name)

    def fit_transform(
            self, tables=None, transformer_dict=None, transformer_list=None, missing=True):
        """Create, apply and store the specified transformers for the given tables.

        Args:
            tables(dict):   Mapping of table names to `tuple` where each tuple is on the form
                            (`pandas.DataFrame`, `dict`). The `DataFrame` contains the table data
                            and the `dict` the corresponding meta information.
                            If not specified, the tables will be retrieved using the meta_file.

            transformer_dict(dict):     Mapping  `tuple(str, str)` -> `str` where the tuple is
                                        (table_name, column_name).

            transformer_list(list):     List of transformers to use. Overrides the transformers in
                                        the meta_file.

            missing(bool): Wheter or not use NullTransformer to handle missing values.

        Returns:
            dict: Map from `str` (table_names) to `pandas.DataFrame` (transformed data).
        """
        transformed = {}

        if tables is None:
            tables = self.table_dict

        if transformer_dict is None and transformer_list is None:
            transformer_dict = self.transformer_dict

        for table_name in tables:
            table, table_meta = tables[table_name]
            transformed_table = self.fit_transform_table(
                table, table_meta, transformer_dict, transformer_list, missing)

            transformed[table_name] = transformed_table

        return transformed

    def transform(self, tables, table_metas=None, missing=True):
        """Apply all the saved transformers to `tables`.

        Args:
            tables(dict):   mapping of table names to `tuple` where each tuple is on the form
                            (`pandas.DataFrame`, `dict`). The `DataFrame` contains the table data
                            and the `dict` the corresponding meta information.
                            If not specified, the tables will be retrieved using the meta_file.

            table_metas(dict):  Full metadata file for the dataset.

            missing(bool): Wheter or not use NullTransformer to handle missing values.

        Returns:
            dict: Map from `str` (table_names) to `pandas.DataFrame` (transformed data).
        """
        transformed = {}

        for table_name in tables:
            table = tables[table_name]

            if table_metas is None:
                table_meta = self.table_dict[table_name][1]
            else:
                table_meta = table_metas[table_name]

            transformed[table_name] = self.transform_table(table, table_meta, missing)

        return transformed

    def reverse_transform(self, tables, table_metas=None, missing=True):
        """Transform data back to its original format.

        Args:
            tables(dict):   mapping of table names to `tuple` where each tuple is on the form
                            (`pandas.DataFrame`, `dict`). The `DataFrame` contains the transformed
                            data and the `dict` the corresponding meta information.
                            If not specified, the tables will be retrieved using the meta_file.

            table_metas(dict):  Full metadata file for the dataset.

            missing(bool): Wheter or not use NullTransformer to handle missing values.

        Returns:
            dict: Map from `str` (table_names) to `pandas.DataFrame` (transformed data).
        """
        reverse = {}

        for table_name in tables:
            table = tables[table_name]
            if table_metas is None:
                table_meta = self.table_dict[table_name][1]
            else:
                table_meta = table_metas[table_name]

            reverse[table_name] = self.reverse_transform_table(table, table_meta, missing)

        return reverse

    def fit_transform_table(
            self, table, table_meta, transformer_dict=None, transformer_list=None, missing=True):
        """Create, apply and store the specified transformers for `table`.

        Args:
            table(pandas.DataFrame):    Contents of the table to be transformed.

            table_meta(dict):   Metadata for the given table.

            transformer_dict(dict):     Mapping  `tuple(str, str)` -> `str` where the tuple is
                                        (table_name, column_name).

            transformer_list(list):     List of transformers to use. Overrides the transformers in
                                        the meta_file.

            missing(bool):      Wheter or not use NullTransformer to handle missing values.

        Returns:
            pandas.DataFrame: Transformed table.
        """
        out = pd.DataFrame(columns=[])
        table_name = table_meta['name']
        for field in table_meta['fields']:
            col_name = field['name']
            col = table[col_name]

            if transformer_list is not None:
                for transformer_name in transformer_list:
                    transformer = self.get_class(transformer_name)
                    t = transformer()
                    if field['type'] == t.type:
                        new_col = t.fit_transform(col, field, missing)
                        # handle missing
                        # if missing:
                        #     null_t = null_class()
                        #     new_col = null_t.fit_transform(new_col, field)
                        self.transformers[(table_name, col_name)] = t
                        out = pd.concat([out, new_col], axis=1)

            elif (table_name, col_name) in transformer_dict:
                transformer_name = transformer_dict[(table_name, col_name)]
                transformer = self.get_class(TRANSFORMERS[transformer_name])
                t = transformer()
                new_col = t.fit_transform(col, field, missing)
                self.transformers[(table_name, col_name)] = t
                out = pd.concat([out, new_col], axis=1)

        return out

    def transform_table(self, table, table_meta):
        """Apply the stored transformers to `table`.

        Args:
            table(pandas.DataFrame):     Contents of the table to be transformed.

            table_meta(dict):   Metadata for the given table.

        Returns:
            pandas.DataFrame: Transformed table.
        """
        out = pd.DataFrame(columns=[])
        table_name = table_meta['name']

        for field in table_meta['fields']:
            col_name = field['name']
            col = table[col_name]
            transformer = self.transformers[(table_name, col_name)]
            out = pd.concat([out, transformer.transform(col, field)], axis=1)

        return out

    def reverse_transform_table(self, table, table_meta, missing=True):
        """Transform a `table` back to its original format.

        Args:
            table(pandas.DataFrame):     Contents of the table to be transformed.

            table_meta(dict):   Metadata for the given table.

        Returns:
            pandas.DataFrame: Table in original format.
        """
        # to check for missing value class
        out = pd.DataFrame(columns=[])
        table_name = table_meta['name']

        for field in table_meta['fields']:
            col_name = field['name']

            # only add transformed columns
            if col_name not in table:
                continue

            col = table[col_name]
            if (table_name, col_name) in self.transformers:
                transformer = self.transformers[(table_name, col_name)]

                if missing:
                    missing_col = table['?' + col_name]
                    data = pd.concat([col, missing_col], axis=1)
                    out_list = [out, transformer.reverse_transform(data, field, missing)]

                else:
                    new_col = transformer.reverse_transform(col, field, missing)
                    out_list = [out, new_col]

                out = pd.concat(out_list, axis=1)

        return out
