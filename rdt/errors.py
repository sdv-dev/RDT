"""RDT Exceptions."""


class ConfigNotSetError(Exception):
    """Error to use when no config has been set or detected."""


class InvalidConfigError(Exception):
    """Error to raise when something is incorrect about the config."""


class InvalidDataError(Exception):
    """Error to raise when the data is ill-formed in some way."""


class NotFittedError(Exception):
    """Error to raise when ``transform`` or ``reverse_transform`` are used before fitting."""


class TransformerInputError(Exception):
    """Error to raise when ``HyperTransformer`` receives an incorrect input."""


class TransformerProcessingError(Exception):
    """Error to raise when transformer fails to complete some process (ie. anonymization)."""
