from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd

from rdt.transformers import IdentityTransformer


class TestIdentityTransformer(TestCase):

    def test___init__(self):
        instance = IdentityTransformer()
        assert instance.INPUT_TYPE is None
        assert instance.OUTPUT_TYPES is None

    def test_fit(self):
        # setup
        data = pd.DataFrame({
            'a': np.array([1, 2, 3]),
            'b': np.array(['a', 'b', 'c']),
            'c': np.array([1., 2., 3.])
        })

        _fit_mock = Mock()
        instance = IdentityTransformer()
        instance._fit = _fit_mock

        instance.INPUT_TYPE = {'a': 'int', 'b': 'object'}
        instance.OUTPUT_TYPES = {'a': 'int', 'b': 'object'}

        # run
        instance.fit(data, columns=['a', 'b'])

        # assert
        _fit_data_arg = _fit_mock.call_args[0][0]
        pd.testing.assert_frame_equal(data[['a', 'b']], _fit_data_arg)

    def test__fit(self):
        # setup
        data = pd.DataFrame({
            'a': np.array([1, 2, 3]),
            'b': np.array(['a', 'b', 'c']),
            'c': np.array([1., 2., 3.])
        })
        instance = IdentityTransformer()

        # run
        instance.fit(data, columns=['a', 'b'])

        # assert
        dtypes = data.dtypes[['a', 'b']]

        instance.INPUT_TYPE == dict(dtypes)
        instance.OUTPUT_TYPES == dict(dtypes)

    def test_transform(self):
        data = pd.DataFrame({
            'a': np.array([1, 2, 3]),
            'b': np.array(['a', 'b', 'c']),
            'c': np.array([1., 2., 3.])
        })

        instance = IdentityTransformer()
        transformed = instance.transform(data)

        # assert
        pd.testing.assert_frame_equal(data, transformed)

    def test_reverse_transform(self):
        data = pd.DataFrame({
            'a': np.array([1, 2, 3]),
            'b': np.array(['a', 'b', 'c']),
            'c': np.array([1., 2., 3.])
        })

        instance = IdentityTransformer()
        reverse_transformed = instance.reverse_transform(data)

        # assert
        pd.testing.assert_frame_equal(data, reverse_transformed)
