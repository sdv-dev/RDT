import pandas as pd

from rdt import transformers, utils

TRANSFORMERS = {
    'datetime': 'DTTransformer',
    'categorical': 'CatTransformer',
    'number': 'NumberTransformer'
}


class HyperTransformer(object):
    """ This class is responsible for formatting the input table in a way
    that is machine learning friendly
    """

    def __init__(self, meta_file=None):
        """Initialize HyperTransformer."""

        self.transformers = {}  # key=(table_name, col_name) val=transformer

        if not meta_file:
            raise ValueError("'meta_file' argument is needed to use HyperTransformer")

        self.table_dict = utils.get_table_dict(meta_file)
        self.transformer_dict = utils.get_transformers_dict(meta_file)

    @staticmethod
    def get_class(class_name):
        """Gets class object of transformer from class name."""
        return getattr(transformers, class_name)

    def fit_transform(
            self, tables=None, transformer_dict=None, transformer_list=None, missing=True):
        """
        This function loops applies all the specified transformers to the
        tables and return a dict of transformed tables

        :param tables: mapping of table names to tuple of (tables to be
        transformed, corresponding table meta). If not specified,
        the tables will be retrieved using the meta_file.

        :param transformer_list: list of transformers to use. If not
        specified, the transformers in the meta_file will be used.

        :returns: dict mapping table name to transformed tables
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
        """
        This function applies all the saved transformers to the
        tables and returns a dict of transformed tables

        :param tables: mapping of table names to tables to be transformed.
        If not specified, the tables will be retrieved using the meta_file.

        :returns: dict mapping table name to transformed tables
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
        """Loops through the list of reverse transform functions and puts data
        back into original format.

        :param tables: mapping of table names to tables to be transformed.
        If not specified, the tables will be retrieved using the meta_file.

        :returns: dict mapping table name to transformed tables"""
        reverse = {}

        for table_name in tables:
            table = tables[table_name]
            if table_metas is None:
                table_meta = self.table_dict[table_name][1]
            else:
                table_meta = table_metas[table_name]

            reverse[table_name] = self.reverse_transform_table(table, table_meta, missing)
        return reverse

    def fit_transform_table(self, table, table_meta, transformer_dict=None,
                            transformer_list=None, missing=True):
        """ Returns the processed table after going through each transform
        and adds fitted transformers to the hyper class
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
            else:
                # use transformer dict
                if (table_name, col_name) in transformer_dict:
                    transformer_name = transformer_dict[(table_name, col_name)]
                    transformer = self.get_class(TRANSFORMERS[transformer_name])
                    t = transformer()
                    new_col = t.fit_transform(col, field, missing)
                    self.transformers[(table_name, col_name)] = t
                    out = pd.concat([out, new_col], axis=1)
        return out

    def transform_table(self, table, table_meta):
        """ Does the required transformations to the table """
        out = pd.DataFrame(columns=[])
        table_name = table_meta['name']

        for field in table_meta['fields']:
            col_name = field['name']
            col = table[col_name]
            transformer = self.transformers[(table_name, col_name)]
            out = pd.concat([out, transformer.transform(col, field)], axis=1)

        return out

    def reverse_transform_table(self, table, table_meta, missing=True):
        """ Converts data back into original format by looping
        over all transformers and doing the reverse
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

    def impute_table(self, table):
        """ Fills in any NaN values in a table """
        values = {}

        for label in table:
            if not pd.isnull(table[label].mean()):
                values[label] = table[label].mean()
            else:
                values[label] = 0

        imputed_table = table.fillna(values)

        return imputed_table

    def get_types(self, table):
        """ Maps every field name to a type """
        res = {}

        for field in table['fields']:
            res[field['name']] = field['type']

        return res
