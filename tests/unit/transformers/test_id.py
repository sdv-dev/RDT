import numpy as np
import pandas as pd

from rdt.transformers.id import IDGenerator


class TestIDGenerator:

    def test___init__default(self):
        """Test the ``__init__`` method."""
        # Run
        transformer = IDGenerator()

        # Assert
        assert transformer.prefix is None
        assert transformer.starting_value == 0
        assert transformer.suffix is None
        assert transformer.counter == 0
        assert transformer.output_properties == {None: {'next_transformer': None}}

    def test___init__with_parameters(self):
        """Test the ``__init__`` method with paremeters."""
        # Run
        transformer_prefix = IDGenerator(prefix='prefix_')
        transformer_suffix = IDGenerator(suffix='_suffix')
        transformer_starting_value = IDGenerator(starting_value=10)
        transformer_all = IDGenerator(prefix='prefix_', starting_value=10, suffix='_suffix')

        # Assert
        assert transformer_prefix.prefix == 'prefix_'
        assert transformer_prefix.starting_value == 0
        assert transformer_prefix.suffix is None
        assert transformer_prefix.counter == 0
        assert transformer_prefix.output_properties == {None: {'next_transformer': None}}

        assert transformer_suffix.prefix is None
        assert transformer_suffix.starting_value == 0
        assert transformer_suffix.suffix == '_suffix'
        assert transformer_suffix.counter == 0
        assert transformer_suffix.output_properties == {None: {'next_transformer': None}}

        assert transformer_starting_value.prefix is None
        assert transformer_starting_value.starting_value == 10
        assert transformer_starting_value.suffix is None
        assert transformer_starting_value.counter == 0
        assert transformer_starting_value.output_properties == {None: {'next_transformer': None}}

        assert transformer_all.prefix == 'prefix_'
        assert transformer_all.starting_value == 10
        assert transformer_all.suffix == '_suffix'
        assert transformer_all.counter == 0
        assert transformer_all.output_properties == {None: {'next_transformer': None}}

    def test_reset_sampling(self):
        """Test the ``reset_sampling`` method."""
        # Setup
        transformer = IDGenerator()
        transformer.counter = 10

        # Run
        transformer.reset_sampling()

        # Assert
        assert transformer.counter == 0

    def test__fit(self):
        """Test the ``_fit`` method."""
        # Setup
        transformer = IDGenerator()

        # Run
        transformer._fit(None)

        # Assert
        assert True

    def test__transform(self):
        """Test the ``_transform`` method."""
        # Setup
        transformer = IDGenerator()

        # Run
        result = transformer._transform(None)

        # Assert
        assert result is None

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method."""
        # Setup
        transformer = IDGenerator()
        transformer.counter = 10

        # Run
        result = transformer._reverse_transform(np.array([1, 2, 3]))

        # Assert
        assert isinstance(result, pd.Series)
        assert result.tolist() == ['10', '11', '12']
        assert transformer.counter == 13

    def test__reverse_transform_with_everything(self):
        """Test the ``_reverse_transform`` method with all parameters."""
        # Setup
        transformer = IDGenerator(prefix='prefix_', starting_value=100, suffix='_suffix')

        # Run
        result = transformer._reverse_transform(np.array([1, 2, 3]))

        # Assert
        assert isinstance(result, pd.Series)
        assert result.tolist() == ['prefix_100_suffix', 'prefix_101_suffix', 'prefix_102_suffix']
        assert transformer.counter == 3
