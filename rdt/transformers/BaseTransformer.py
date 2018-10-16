class BaseTransformer(object):
    """Base class for all transformers."""

    def __init__(self, col_meta=None, missing=True, type=None):
        """Initialize preprocessor.

        Args:
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.
            transformer_type(str): Type of data the transformer is able to transform.
        """
        self.type = type
        self.col_name = None
        self.col_meta = col_meta
        self.missing = missing

    def fit(self, col, col_meta=None, missing=None):
        """Prepare the transformer to convert data.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError

    def transform(self, col, col_meta=None, missing=None):
        """Does the required transformations to the data.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        return self.fit_transform(col, col_meta, missing)

    def fit_transform(self, col, col_meta=None, missing=None):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        self.fit(col, col_meta, missing)
        return self.transform(col, col_meta, missing)

    def reverse_transform(self, col, col_meta=None, missing=None):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError

    def check_data_type(self, col_meta):
        """Check the type of the transformer and column match.

        Args:
            col_meta(dict): Metadata of the column.

        Raises a ValueError if the types don't match
        """

        if self.type != col_meta.get('type'):
            raise ValueError('Types of transformer don\'t match')
