import argparse
import sys


def test_performance(transformers, sizes):
    """
    Function to test the performance of specified transformers.

    This function will loop through the provided transformers
    and dataset sizes and use that information to determine which
    test cases to run. It will then execute those tests.

    Args:
        transformers (list):
            A list of transformer classes to test.
        sizes (list):
            A list of sizes to use for the generated datasets. Each
            size should be an int of how many rows that dataset should
            have.
    """
    pass


if __name__ == '__main__':
    # everything below is just experimenting
    parser = argparse.ArgumentParser(description='Test RDT Performance')
    parser.add_argument('transformers', nargs='+', help='List of transformer class names.')
    parser.add_argument('sizes', nargs='+', help='List of column lengths.')

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
