import numpy as np
import pandas as pd

from rdt.transformers import BooleanTransformer


class TestBooleanTransformer:

    def test_boolean_some_nans(self):
        """Test BooleanTransformer on input with some nan values.

        Ensure that the BooleanTransformer can fit, transform, and reverse
        transform on boolean data with Nones. Expect that the reverse
        transformed data is the same as the input.

        Input:
            - boolean data with None values
        Output:
            - The reversed transformed data
        """
        # Setup
        data = pd.DataFrame([True, False, None, False], columns=['bool'])
        transformer = BooleanTransformer()

        # Run
        transformer.fit(data, data.columns.to_list())
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(reverse, data)

    def test_boolean_all_nans(self):
        """Test BooleanTransformer on input with all nan values.

        Ensure that the BooleanTransformer can fit, transform, and reverse
        transform on boolean data with all Nones. Expect that the reverse
        transformed data is the same as the input.

        Input:
            - 4 rows of all None values
        Output:
            - The reversed transformed data
        """
        # Setup
        data = pd.DataFrame([None, None, None, None], columns=['bool'])
        transformer = BooleanTransformer()

        # Run
        transformer.fit(data, data.columns.to_list())
        transformed = transformer.transform(data)
        reverse = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(reverse, data)

    def test_boolean_input_unchanged(self):
        """Test BooleanTransformer on input with some nan values.

        Ensure that the BooleanTransformer can fit, transform, and reverse
        transform on boolean data with all Nones. Expect that the intermediate
        transformed data is unchanged.

        Input:
            - 4 rows of all None values
        Output:
            - The reversed transformed data
        Side effects:
            - The intermediate transformed data is unchanged.
        """
        # Setup
        data = pd.DataFrame([True, False, None, False], columns=['bool'])
        transformer = BooleanTransformer()

        # Run
        transformer.fit(data, data.columns.to_list())
        transformed = transformer.transform(data)
        unchanged_transformed = transformed.copy()
        reverse = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_frame_equal(reverse, data)
        np.testing.assert_array_equal(unchanged_transformed, transformed)
