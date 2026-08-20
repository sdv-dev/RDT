"""Transformers for text data."""

import warnings

from rdt.transformers.id import RegexGenerator  # noqa: F401

warnings.warn(
    "Importing 'RegexGenerator' for ID columns from 'rdt.transformers.text' "
    "is deprecated. Please use 'rdt.transformers.id' instead.",
    DeprecationWarning,
    stacklevel=2,
)
