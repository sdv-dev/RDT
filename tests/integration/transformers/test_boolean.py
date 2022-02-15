import numpy as np
import pandas as pd

from rdt.transformers import BinaryEncoder


class TestBinaryEncoder:

    def test_boolean_some_nans(self):
        """Test BinaryEncoder on input with some nan values.

        Ensure that the BinaryEncoder can fit, transform, and reverse
        transform on boolean data with Nones. Expect that the reverse
        transformed data is the same as the input.

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
        reverse = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(reverse, data)

    def test_boolean_all_nans(self):
        """Test BinaryEncoder on input with all nan values.

        Ensure that the BinaryEncoder can fit, transform, and reverse
        transform on boolean data with all Nones. Expect that the reverse
        transformed data is the same as the input.

        Input:
            - 4 rows of all None values
        Output:
            - The reversed transformed data
        """
        # Setup
        data = pd.DataFrame([None, None, None, None], columns=['bool'])
        column = 'bool'
        transformer = BinaryEncoder()

        # Run
        transformer.fit(data, column)
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(reverse, data)

    def test_boolean_input_unchanged(self):
        """Test BinaryEncoder doesn't affect transformed data.

        Ensure that the intermediate transformed data is unchanged after reverse transforming.

        Input:
            - boolean data
        Output:
            - The reversed transformed data
        Side effects:
            - The intermediate transformed data is unchanged.
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
        pd.testing.assert_frame_equal(reverse, data)
        np.testing.assert_array_equal(unchanged_transformed, transformed)

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
            'bool.value': [1., 1., 1., 0.],
            'bool.is_null': [0., 0., 1., 0.]
        })
        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse, data)
