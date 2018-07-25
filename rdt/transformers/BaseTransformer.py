class BaseTransformer(object):
    """ This class is responsible for formatting the input table in
    a way that is machine learning friendly
    """

    def __init__(self):
        """ initialize preprocessor """
        pass

    def fit_transform(self, col, col_meta, missing=True):
        """ Returns the processed table """
        raise NotImplementedError

    def transform(self, col, col_meta, missing=True):
        """ Does the required transformations to the data """
        return self.fit_transform(col, col_meta, missing)

    def reverse_transform(self, col, col_meta, missing=True):
        """ Converts data back into original format """
        raise NotImplementedError
