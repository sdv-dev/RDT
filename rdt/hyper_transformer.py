import json
import os
import warnings

import pandas as pd

from rdt import transformers

TRANSFORMERS = {
    'datetime': 'DTTransformer',
    'categorical': 'CatTransformer',
    'number': 'NumberTransformer'
}


DEPRECATION_MESSAGE = (
    'Usage of the argument `missing` in the method `{}` is soon to be deprecated '
    'and should be used at class-level. It will stop working after release v0.2.0.'
)


class HyperTransformer:
    """Multitable transformer.

    The main propouse of the HyperTransformer class is to easily manage transformations
    for a whole dataset.

    Args:
        metadata (str or dict):
            Path to the meta.json file or its parsed contents.
        dir_name (str):
            Path to the root directory of meta.json. Defaults to ``None``.
        missing (bool):
            Wheter or not to handle missing values before transforming data.
            Defaults to ``True``.

    Raises:
        ValueError:
            A ``ValueError`` is raised when the ``metadata`` is a ``dict`` and ``dir_name``
            is ``None`` or when the instance of ``metadata`` is not a ``str`` or a ``dict``.
    """

    @staticmethod
    def get_class(class_name):
        """Get class object of transformer from its class name.

        Args:
            class_name (str):
                Name of the transform.

        Returns:
            BaseTransformer
        """
        return getattr(transformers, class_name)

    @staticmethod
    def _get_pii_fields(table_metadata):
        """Return a list of fields marked as sensitive information.

        Args:
            table_metadata (dict):
                Metadata corresponding to a table.

        Returns:
            list[dict]:
                List of metadata for each field marked as ``pii``.
        """
        return [field for field in table_metadata['fields'] if field.get('pii')]

    @classmethod
    def _anonymize_table(cls, table_data, pii_fields):
        """Anonymize in ``table_data`` the fields in ``pii_fields``.

        Args:
            table_data (pandas.DataFrame):
                Original dataframe/table.
            pii_fields (list[dict]):
                Metadata for the fields to transform.

        Returns:
            pandas.DataFrame:
                Anonymized table.
        """
        for pii_field in pii_fields:
            field_name = pii_field['name']
            transformer = cls.get_class(TRANSFORMERS['categorical'])(pii_field)
            table_data[field_name] = transformer.anonymize_column(table_data)

        return table_data

    def _get_tables(self, base_dir):
        """Load the contents of meta_file and the corresponding data.

        If fields containing Personally Identifiable Information are detected in the metadata
        they are anonymized before asign them into ``table_dict``.

        Args:
            base_dir (str):
                Root folder of the dataset files.

        Returns:
            dict:
                Mapping str -> tuple(pandas.DataFrame, dict)
        """
        table_dict = {}

        for table in self.metadata['tables']:
            if table['use']:
                relative_path = os.path.join(base_dir, self.metadata['path'], table['path'])
                data_table = pd.read_csv(relative_path)
                pii_fields = self._get_pii_fields(table)
                data_table = self._anonymize_table(data_table, pii_fields)

                table_dict[table['name']] = (data_table, table)

        return table_dict

    def _get_transformers(self):
        """Load the contents of meta_file and extract information about the transformers.

        Returns:
            dict:
                tuple(str, str) -> Transformer.
        """
        transformer_dict = {}

        for table in self.metadata['tables']:
            table_name = table['name']

            for field in table['fields']:
                transformer_type = field.get('type')
                if transformer_type:
                    col_name = field['name']
                    transformer_dict[(table_name, col_name)] = transformer_type

        return transformer_dict

    def __init__(self, metadata, dir_name=None, missing=True):

        self.transformers = {}  # key=(table_name, col_name) val=transformer

        if isinstance(metadata, str):
            dir_name = os.path.dirname(metadata)
            with open(metadata, 'r') as f:
                self.metadata = json.load(f)

        elif isinstance(metadata, dict):
            self.metadata = metadata
            if dir_name is None:
                raise ValueError('dir_name is required when metadata is a dict.')

        else:
            raise ValueError('Incorrect type for metadata: It can only be either dict or str.')

        self.table_dict = self._get_tables(dir_name)
        self.transformer_dict = self._get_transformers()
        self.missing = missing

    def _fit_transform_column(self, table, metadata, transformer_name, table_name):
        """Transform a column from ``table`` using transformer and given parameters.

        Args:
            table (pandas.DataFrame):
                Dataframe containing column to transform.
            metadata (dict):
                Metadata for given column.
            transformer_name (str):
                Name of transformer to use on column.
            table_name (str):
                Name of table in original dataset.

        Returns:
            pandas.DataFrame:
                Dataframe containing the transformed column. If ``self.missing=True``,
                it will contain a second column containing 0 and 1 marking if that
                value was originally null or not.
        """

        column_name = metadata['name']
        content = {}
        columns = []

        if self.missing and table[column_name].isnull().any():
            null_transformer = transformers.NullTransformer(metadata)
            clean_column = null_transformer.fit_transform(table[column_name])
            null_name = '?' + column_name
            columns.append(null_name)
            content[null_name] = clean_column[null_name].values
            table[column_name] = clean_column[column_name]

        transformer_class = self.get_class(transformer_name)
        transformer = transformer_class(metadata)

        self.transformers[(table_name, column_name)] = transformer
        content[column_name] = transformer.fit_transform(table)[column_name].values

        columns = [column_name] + columns
        return pd.DataFrame(content, columns=columns)

    def _reverse_transform_column(self, table, metadata, table_name):
        """Reverses the transformation on a column from ``table`` using the given parameters.

        Args:
            table (pandas.DataFrame):
                Dataframe containing column to transform.
            metadata (dict):
                Metadata for given column.
            table_name (str):
                Name of the table in the original dataset.

        Returns:
            pandas.DataFrame:
                Dataframe containing the transformed column. If ``self.missing=True``,
                it will contain a second column containing 0 and 1 marking if that
                value was originally null or not.
                It will return ``None`` in the case the column is not in the table.
        """

        column_name = metadata['name']

        if column_name not in table:
            return

        null_name = '?' + column_name
        content = pd.DataFrame(columns=[column_name], index=table.index)
        transformer = self.transformers[(table_name, column_name)]
        content[column_name] = transformer.reverse_transform(table[column_name].to_frame())

        if self.missing and null_name in table[column_name]:
            content[null_name] = table.pop(null_name)
            null_transformer = transformers.NullTransformer(metadata)
            content[column_name] = null_transformer.reverse_transform(content)

        return content

    def fit_transform_table(self, table, table_meta, transformer_dict=None,
                            transformer_list=None, missing=None):
        """Create, apply and store the specified transformers for ``table``.

        Args:
            table (pandas.DataFrame):
                Contents of the table to be transformed.
            table_meta (dict):
                Metadata for the given table.
            transformer_dict (dict):
                Mapping ``tuple(str, str)`` -> ``str`` where the tuple in the keys represent
                the ``(table_name, column_name)`` and the value the name
                of the assigned transformer. Defaults to ``None``.
            transformer_list (list):
                List of transformers to use. Overrides the transformers in the ``meta_file``.
                Defaults to ``None``.
            missing (bool):
                Wheter or not use ``NullTransformer`` to handle missing values.
                Defaults to ``None``

        Returns:
            pandas.DataFrame:
                Transformed table.
        """

        if missing is not None:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('fit_transform_table'), DeprecationWarning)

        result = pd.DataFrame()
        table_name = table_meta['name']

        for field in table_meta['fields']:
            col_name = field['name']

            if transformer_list:
                for transformer_name in transformer_list:
                    if field['type'] == self.get_class(transformer_name).type:
                        transformed = self._fit_transform_column(
                            table, field, transformer_name, table_name)

                        result = pd.concat([result, transformed], axis=1)

            elif (table_name, col_name) in transformer_dict:
                transformer_name = TRANSFORMERS[transformer_dict[(table_name, col_name)]]
                transformed = self._fit_transform_column(
                    table, field, transformer_name, table_name)

                result = pd.concat([result, transformed], axis=1)

        return result

    def transform_table(self, table, table_meta, missing=None):
        """Apply the stored transformers to ``table``.

        Args:
            table (pandas.DataFrame):
                Contents of the table to be transformed.
            table_meta (dict):
                Metadata for the given table.
            missing (bool):
                Wheter or not use ``NullTransformer`` to handle missing values.
                Defaults to ``None``

        Returns:
            pandas.DataFrame:
                Transformed table.
        """

        if missing is not None:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('transform_table'), DeprecationWarning)

        content = {}
        columns = []
        table_name = table_meta['name']

        for field in table_meta['fields']:
            column_name = field['name']

            if missing and table[column_name].isnull().any():
                null_transformer = transformers.NullTransformer(field)
                clean_column = null_transformer.fit_transform(table[column_name])
                null_name = '?' + column_name
                columns.append(null_name)
                content[null_name] = clean_column[null_name].values
                column = clean_column[column_name]

            else:
                column = table[column_name].to_frame()

            transformer = self.transformers[(table_name, column_name)]
            content[column_name] = transformer.transform(column)[column_name].values
            columns.append(column_name)

        return pd.DataFrame(content, columns=columns)

    def reverse_transform_table(self, table, table_meta, missing=None):
        """Transform a ``table`` back to its original format.

        Args:
            table (pandas.DataFrame):
                Contents of the table to be transformed.
            table_meta (dict):
                Metadata for the given table.
            missing (bool):
                Wheter or not use ``NullTransformer`` to handle missing values.
                Defaults to ``None``

        Returns:
            pandas.DataFrame:
                Table in original format.
        """

        if missing is not None:
            self.missing = missing
            warnings.warn(
                DEPRECATION_MESSAGE.format('reverse_transform_table'), DeprecationWarning)

        result = pd.DataFrame(index=table.index)
        table_name = table_meta['name']

        for field in table_meta['fields']:
            new_column = self._reverse_transform_column(table, field, table_name)
            if new_column is not None:
                result[field['name']] = new_column

        return result

    def fit_transform(self, tables=None, transformer_dict=None,
                      transformer_list=None, missing=None):
        """Create, apply and store the specified transformers for the given tables.

        Args:
            tables (dict):
                Mapping of table names to ``tuple`` where each tuple is on the form
                ``(pandas.DataFrame, dict)``. The ``DataFrame`` contains the table data
                and the ``dict`` the corresponding meta information.
                If not specified, the tables will be retrieved using the meta_file.
                Defaults to ``None``.
            transformer_dict (dict):
                Mapping ``tuple(str, str)`` -> ``str`` where the tuple is
                ``(table_name, column_name)``. Defaults to ``None``.
            transformer_list (list):
                List of transformers to use. Overrides the transformers in the ``meta_file``.
                Defaults to ``None``.
            missing (bool):
                Wheter or not use ``NullTransformer`` to handle missing values.
                Defaults to ``None``.

        Returns:
            dict:
                Map from ``str`` (table_names) to ``pandas.DataFrame`` (transformed data).
        """

        if missing is not None:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('fit_transform'), DeprecationWarning)

        transformed = {}

        if tables is None:
            tables = self.table_dict

        if transformer_dict is None and transformer_list is None:
            transformer_dict = self.transformer_dict

        for table_name in tables:
            table, table_meta = tables[table_name]
            transformed_table = self.fit_transform_table(
                table, table_meta, transformer_dict, transformer_list)

            transformed[table_name] = transformed_table

        return transformed

    def transform(self, tables, table_metas=None, missing=None):
        """Apply all the saved transformers to ``tables``.

        Args:
            tables(dict):
                mapping of table names to ``tuple`` where each tuple is on the form
                (``pandas.DataFrame``, ``dict``). The ``DataFrame`` contains the table data
                and the ``dict`` the corresponding meta information.
                If not specified, the tables will be retrieved using the meta_file.
            table_metas(dict):
                Full metadata file for the dataset. Defaults to ``None``.
            missing(bool):
                Wheter or not use ``NullTransformer`` to handle missing values.
                Defaults to ``None``.

        Returns:
            dict:
                Map from ``str`` (table_names) to ``pandas.DataFrame`` (transformed data).
        """

        if missing is not None:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('transform'), DeprecationWarning)

        transformed = {}

        for table_name in tables:
            table = tables[table_name]

            if table_metas is None:
                table_meta = self.table_dict[table_name][1]
            else:
                table_meta = table_metas[table_name]

            transformed[table_name] = self.transform_table(table, table_meta)

        return transformed

    def reverse_transform(self, tables, table_metas=None, missing=None):
        """Transform data back to its original format.

        Args:
            tables(dict):
                mapping of table names to ``tuple`` where each tuple is on the form
                (``pandas.DataFrame``, ``dict``). The ``DataFrame`` contains the transformed
                data and the ``dict`` the corresponding meta information.
                If not specified, the tables will be retrieved using the meta_file.
            table_metas(dict):
                Full metadata file for the dataset. Defaults to ``None``.
            missing(bool):
                Wheter or not use ``NullTransformer`` to handle missing values.
                Defaults to ``None``

        Returns:
            dict:
                Map from ``str`` (table_names) to ``pandas.DataFrame`` (transformed data).
        """

        if missing is not None:
            self.missing = missing
            warnings.warn(DEPRECATION_MESSAGE.format('reverse_transform'), DeprecationWarning)

        reverse = {}

        for table_name in tables:
            table = tables[table_name]
            if table_metas is None:
                table_meta = self.table_dict[table_name][1]
            else:
                table_meta = table_metas[table_name]

            reverse[table_name] = self.reverse_transform_table(table, table_meta)

        return reverse
