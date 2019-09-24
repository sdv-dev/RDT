class BaseTransformer(object):
    """Base class for all transformers."""

    type = None

    def __init__(self, column_metadata):
        """Initialize preprocessor.

        Args:
            column_metadata (dict):
                Meta information of the column.
            transformer_type (str):
                Type of data the transformer is able to transform.
        """
        self.column_metadata = column_metadata
        self.col_name = column_metadata['name']

        self.check_data_type()

    def fit(self, col):
        """Prepare the transformer to convert data.

        Args:
            col (pandas.DataFrame):
                Data to transform.
        """
        raise NotImplementedError

    def transform(self, col):
        """Does the required transformations to the data.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError

    def fit_transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        self.fit(col)
        return self.transform(col)

    def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError

    def check_data_type(self):
        """Check the type of the transformer and column match.

        Args:
            column_metadata (dict):
                Metadata of the column.

        Raises:
            ValueError:
                A ``ValueError`` is raised if the types don't match.
        """
        metadata_type = self.column_metadata.get('type')
        if self.type != metadata_type and metadata_type not in self.type:
            raise ValueError('Types of transformer don\'t match')
