import pandas as pd

from tests.datasets import boolean

NUM_ROWS = 50


class TestRandomBooleanGenerator:

    def test_generate(self):
        """Test the `RandomBooleanGenerator.generate` method.

        Expect that the specified number of rows of booleans is generated, and
        that there are 2 unique values (True and False).

        Input:
            - the number of rows
        Output:
            - a random boolean array of the specified number of rows
        """
        output = boolean.RandomBooleanGenerator.generate(NUM_ROWS)
        assert len(output) == NUM_ROWS
        assert output.dtype == bool
        assert len(pd.unique(output)) == 2
        assert pd.isna(output).sum() == 0


class TestRandomBooleanNaNsGenerator:

    def test_generate(self):
        """Test the `RandomBooleanNaNsGenerator.generate` method.

        Expect that the specified number of rows of booleans is generated, and
        that there are 3 unique values (True, False, and None).

        Input:
            - the number of rows
        Output:
            - a random boolean array of the specified number of rows, with null values
        """
        output = boolean.RandomBooleanNaNsGenerator.generate(NUM_ROWS)
        assert len(output) == NUM_ROWS
        assert output.dtype == 'O'
        assert len(pd.unique(output)) == 3
        assert pd.isna(output).sum() > 0


class TestRandomSkewedBooleanGenerator:

    def test_generate(self):
        """Test the `RandomSkewedBooleanGenerator.generate` method.

        Expect that the specified number of rows of booleans is generated, and
        that there are 3 unique values (True, False, and None).

        Input:
            - the number of rows
        Output:
            - a skewed random boolean array of the specified number of rows
        """
        output = boolean.RandomSkewedBooleanGenerator.generate(NUM_ROWS)
        assert len(output) == NUM_ROWS
        assert output.dtype == bool
        assert len(pd.unique(output)) == 2
        assert pd.isna(output).sum() == 0


class TestRandomSkewedBooleanNaNsGenerator:

    def test_generate(self):
        """Test the `RandomSkewedBooleanNaNsGenerator.generate` method.

        Expect that the specified number of rows of booleans is generated, and
        that there are 3 unique values (True, False, and None).

        Input:
            - the number of rows
        Output:
            - a skewed random boolean array of the specified number of rows,
              with null values
        """
        output = boolean.RandomSkewedBooleanNaNsGenerator.generate(NUM_ROWS)
        assert len(output) == NUM_ROWS
        assert output.dtype == 'O'
        assert len(pd.unique(output)) == 3
        assert pd.isna(output).sum() > 0


class TestConstantBooleanGenerator:

    def test_generate(self):
        """Test the `ConstantBooleanGenerator.generate` method.

        Expect that the specified number of rows of booleans is generated, and
        that there is only one unique value (True or False).

        Input:
            - the number of rows
        Output:
            - a boolean array of the specified number of rows, with all values equal
              to either True or False
        """
        output = boolean.ConstantBooleanGenerator.generate(NUM_ROWS)
        assert len(output) == NUM_ROWS
        assert output.dtype == bool
        assert len(pd.unique(output)) == 1
        assert pd.isna(output).sum() == 0


class TestConstantBooleanNaNsGenerator:

    def test(self):
        output = boolean.ConstantBooleanNaNsGenerator.generate(NUM_ROWS)
        assert len(output) == NUM_ROWS
        assert output.dtype == 'O'
        assert len(pd.unique(output)) == 2
        assert pd.isna(output).sum() > 0
