import numpy as np
import pandas as pd

from rdt.transformers import IdentityTransformer


class TestIdentityTransformer:

    def test_fit(self):
        data = pd.DataFrame({
            'a': np.array([1, 2, 3]),
            'b': np.array(['a', 'b', 'c']),
            'c': np.array([1., 2., 3.])
        })

        instance = IdentityTransformer()
        instance.fit(data, columns=['a', 'b'])

    def test_transform(self):
        data = pd.DataFrame({
            'a': np.array([1, 2, 3]),
            'b': np.array(['a', 'b', 'c']),
            'c': np.array([1., 2., 3.])
        })

        instance = IdentityTransformer()
        instance.fit(data, columns=['a', 'b'])
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
        instance.fit(data, columns=['a', 'b'])
        reverse_transformed = instance.reverse_transform(data)

        # assert
        pd.testing.assert_frame_equal(data, reverse_transformed)
