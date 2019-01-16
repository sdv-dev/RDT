class BaseTransformer(object):
    """Base class for all transformers."""

    def __init__(self, column_metadata=None, missing=True, type=None):
        """Initialize preprocessor.

        Args:
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.
            transformer_type(str): Type of data the transformer is able to transform.
        """
        self.type = type
        self.col_name = None
        self.column_metadata = column_metadata
        self.missing = missing

    def fit(self, col, column_metadata=None, missing=None):
        """Prepare the transformer to convert data.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.
        """
        raise NotImplementedError

    def transform(self, col, column_metadata=None, missing=None):
        """Does the required transformations to the data.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError

    def fit_transform(self, col, column_metadata=None, missing=None):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        self.fit(col, column_metadata, missing)
        return self.transform(col, column_metadata, missing)

    def reverse_transform(self, col, column_metadata=None, missing=None):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError

    def check_data_type(self, column_metadata):
        """Check the type of the transformer and column match.

        Args:
            column_metadata(dict): Metadata of the column.

        Raises a ValueError if the types don't match
        """

        if self.type != column_metadata.get('type'):
            raise ValueError('Types of transformer don\'t match')
