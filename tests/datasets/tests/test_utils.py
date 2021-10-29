"""Tests for the datasets.utils module."""

import numpy as np

from tests.datasets import utils


def test_add_nulls_int():
    array = np.arange(100)

    with_nans = utils.add_nans(array)

    assert len(with_nans) == 100
    assert 1 <= np.isnan(with_nans).sum() < 100

    nans = np.isnan(with_nans)
    np.testing.assert_array_equal(array[~nans], with_nans[~nans])


def test_add_nulls_float():
    array = np.arange(100).astype(float)

    with_nans = utils.add_nans(array)

    assert len(with_nans) == 100
    assert 1 <= np.isnan(with_nans).sum() < 100

    nans = np.isnan(with_nans)
    np.testing.assert_array_equal(array[~nans], with_nans[~nans])
