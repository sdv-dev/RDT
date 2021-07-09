def profile(transformer, dataset, kwargs):
    """
    Function to profile a transformer.

    This function will get the total time and peak memory
    for the ``fit``, ``transform`` and ``reverse_transform``
    methods of the provided transformer against the provided
    dataset.

    Args:
        transformer (str):
            Transformer class name.
        dataset (numpy.ndarray):
            The dataset to transform.
        kwargs (dict):
            The kwargs used to initialize the transformer.
    """
    pass

if __name__ == '__main__':
    pass