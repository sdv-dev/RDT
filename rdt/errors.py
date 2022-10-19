"""RDT Exceptions."""


class NotFittedError(Exception):
    """Error to raise when ``transform`` or ``reverse_transform`` are used before fitting."""


class Error(Exception):
    """Error to raise when ``HyperTransformer`` produces a controlled error message."""


class TransformerInputError(Exception):
    """Error to raise when ``HyperTransformer`` receives an incorrect input."""
