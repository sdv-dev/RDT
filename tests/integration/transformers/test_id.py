import pandas as pd

from rdt.transformers import IDGenerator


class TestIDGenerator():

    def test_end_to_end(self):
        """End to end test of the ``IDGenerator``."""
        # Setup
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'username': ['a', 'b', 'c', 'd', 'e']
        })

        # Run
        transformer = IDGenerator(prefix='id_', starting_value=100, suffix='_X')
        transformed = transformer.fit_transform(data, 'id')
        reverse_transform = transformer.reverse_transform(transformed)
        reverse_transform_2 = transformer.reverse_transform(transformed)
        transformer.reset_sampling()
        reverse_transform_3 = transformer.reverse_transform(transformed)

        # Assert
        expected_transformed = pd.DataFrame({
            'username': ['a', 'b', 'c', 'd', 'e']
        })

        expected_reverse_transform = pd.DataFrame({
            'username': ['a', 'b', 'c', 'd', 'e'],
            'id': ['id_100_X', 'id_101_X', 'id_102_X', 'id_103_X', 'id_104_X']
        })

        expected_reverse_transform_2 = pd.DataFrame({
            'username': ['a', 'b', 'c', 'd', 'e'],
            'id': ['id_105_X', 'id_106_X', 'id_107_X', 'id_108_X', 'id_109_X']
        })

        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse_transform, expected_reverse_transform)
        pd.testing.assert_frame_equal(reverse_transform_2, expected_reverse_transform_2)
        pd.testing.assert_frame_equal(reverse_transform_3, expected_reverse_transform)
