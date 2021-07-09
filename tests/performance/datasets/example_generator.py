import numpy as np


def example_numerical_generator(num_rows):
    """
    This method serves as a template for dataset generators

    Args:
        num_rows (int):
            Number of rows to generate.
    Returns:
        numpy.ndarray of size ``num_rows``
    """
    ii32 = np.iinfo(np.int32)
    return np.random.randint(ii32.min, ii32.max, num_rows)
