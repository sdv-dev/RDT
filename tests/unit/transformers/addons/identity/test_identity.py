from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from rdt.transformers.addons.identity.identity import IdentityTransformer


class TestIdentityTransformer(TestCase):

    def test__fit(self):
        """Test ``IdentityTransformer._fit`` function.

        The ``_fit`` function is expected to learn the columns ``INPUT_SDTYPE`` and
        ``OUTPUT_COLUMNS`` from the ``self.columns``.

        Setup:
            - mock self.columns to be a list with values.
        Input:
            - ``data``, a ``numpy.ndarray`` with numerical values.
        Output:
            - n/a
        """
        # setup
        instance = Mock()
        instance.columns = ['a', 'b', 'c']

        # run
        data = np.array([0.5, 0.6, 0.7])
        IdentityTransformer._fit(instance, data)

        # assert
        expected_output_properties = {
            'a': {'next_transformer': None},
            'b': {'next_transformer': None},
            'c': {'next_transformer': None},
        }

        assert instance.output_properties == expected_output_properties

    def test__transform(self):
        """Test ``IdentityTransformer._transform`` function.

        The ``_transform`` function is expected to return the input object without modifying it.

        Input:
            - An `object` instance.
        Output:
            - Same ``object`` as input, unmodified.
        """
        # setup
        instance = Mock()
        input_object = object()

        # run
        output_object = IdentityTransformer._transform(instance, input_object)

        # assert
        assert input_object == output_object

    def test__reverse_transform(self):
        """Test ``IdentityTransformer._reverse_transform`` function.

        The ``_reverse_transform`` function is expected to return the input object without
        modifying it.

        Input:
            - An `object` instance.
        Output:
            - Same ``object`` as input, unmodified.
        """
        # setup
        instance = Mock()
        input_object = object()

        # run
        output_object = IdentityTransformer._reverse_transform(instance, input_object)

        # assert
        assert input_object == output_object
