"""Test Text Transformers."""

import pytest


def test_deprecation_warning_is_raised():
    """Test that a deprecation warning is raised when importing from this module."""
    # Run and Assert
    expected_message = (
        "Importing 'IDGenerator' or 'RegexGenerator' for ID columns from 'rdt.transformers.text' "
        "is deprecated. Please use 'rdt.transformers.id' instead."
    )
    with pytest.warns(DeprecationWarning, match=expected_message):
        from rdt.transformers.text import IDGenerator, RegexGenerator  # noqa: F401
