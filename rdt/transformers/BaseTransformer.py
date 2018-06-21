class BaseTransformer(object):
    """ This class is responsible for formatting the input table in
    a way that is machine learning friendly
    """

    def __init__(self):
        """ initialize preprocessor """
        pass

    def fit_transform(self, col, col_meta):
        """ Returns the processed table """
        raise NotImplementedError

    def transform(self, col, col_meta):
        """ Does the required transformations to the data """
        raise NotImplementedError

    def reverse_transform(self, col, col_meta):
        """ Converts data back into original format """
        raise NotImplementedError
