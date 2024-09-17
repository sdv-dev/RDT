"""Transformers for text data."""

import warnings

from rdt.transformers.id import IDGenerator, RegexGenerator  # noqa: F401

warnings.warn(
    "Importing 'IDGenerator' or 'RegexGenerator' for ID columns from 'rdt.transformers.text' "
    "is deprecated. Please use 'rdt.transformers.id' instead.",
    DeprecationWarning,
    stacklevel=2,
)
