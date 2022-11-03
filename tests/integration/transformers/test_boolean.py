import numpy as np
import pandas as pd

from rdt.transformers import BinaryEncoder


class TestBinaryEncoder:

    def test_boolean_some_nans(self):
        """Test BinaryEncoder on input with some nan values.

        Ensure that the BinaryEncoder can fit, transform, and reverse transform on boolean data
        with Nones. Expect that the reverse transformed data is the same as the input, but None
        becomes nan and the False/nan values can be interchanged.

        Also ensures that the intermediate transformed data is unchanged after reversing.

        Input:
            - boolean data with None values
        Output:
            - The reversed transformed data
        """
        # Setup
        data = pd.DataFrame([True, False, None, False], columns=['bool'])
        column = 'bool'
        transformer = BinaryEncoder()

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        unchanged_transformed = transformed.copy()
        reverse = transformer.reverse_transform(transformed)

        # Assert
        np.testing.assert_array_equal(unchanged_transformed, transformed)
        assert reverse['bool'][0] in {True, np.nan}
        for value in reverse['bool'][1:]:
            assert value is False or np.isnan(value)

    def test_boolean_missing_value_replacement_mode(self):
        """Test BinaryEncoder when `missing_value_replacement` is set to 'mode'.

        Ensure that the BinaryEncoder can fit, transform, and reverse transform on
        boolean data when `missing_value_replacement` is set to `'mode'` and
        `model_missing_values` is set to True. Expect that the reverse
        transformed data is the same as the input.

        Input:
            - boolean data with None values
        Output:
            - The reversed transformed data
        """
        # Setup
        data = pd.DataFrame([True, True, None, False], columns=['bool'])
        column = 'bool'
        transformer = BinaryEncoder(missing_value_replacement='mode', model_missing_values=True)

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

        # Assert
        expected_transformed = pd.DataFrame({
            'bool': [1., 1., 1., 0.],
            'bool.is_null': [0., 0., 1., 0.]
        })
        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse, data)
