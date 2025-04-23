import re

import pytest

from rdt._utils import _validate_unique_transformer_instances
from rdt.errors import (
    InvalidConfigError,
)
from rdt.transformers import (
    BaseMultiColumnTransformer,
    BaseTransformer,
)


@pytest.fixture()
def column_name_to_transformer():
    return {
        'colA': BaseTransformer(),
        'colB': BaseTransformer(),
        'colC': BaseTransformer(),
        ('colD', 'colE'): BaseMultiColumnTransformer(),
        ('colF', 'colG'): BaseMultiColumnTransformer(),
        'colH': None,
        'colI': None,
    }


def test__validate_unique_transformer_instances_no_duplicates(column_name_to_transformer):
    """Test the function does not error when no duplicate transformers are present."""
    # Run and Assert
    _validate_unique_transformer_instances(column_name_to_transformer)


def test__validate_unique_transformer_instances_one_duplicate(column_name_to_transformer):
    """Test the function errors when one transformer instance is reused."""
    # Setup
    column_name_to_transformer = column_name_to_transformer.copy()
    column_name_to_transformer['duped_column_1'] = column_name_to_transformer['colA']
    column_name_to_transformer['duped_column_2'] = column_name_to_transformer['colA']

    # Run and Assert
    expected_msg = re.escape(
        "The same transformer instance is being assigned to columns ('colA', 'duped_column_1', "
        "'duped_column_2'). Please create different transformer objects for each assignment."
    )
    with pytest.raises(InvalidConfigError, match=expected_msg):
        _validate_unique_transformer_instances(column_name_to_transformer)


def test__validate_unique_transformer_instances_multi_column(column_name_to_transformer):
    """Test the function with multi-column transformers."""
    # Setup
    column_name_to_transformer = column_name_to_transformer.copy()
    duplicate_transformer = column_name_to_transformer[('colD', 'colE')]
    column_name_to_transformer[('duped_column_1', 'duped_column_2')] = duplicate_transformer

    # Run and Assert
    expected_msg = re.escape(
        "The same transformer instance is being assigned to columns (('colD', 'colE'), "
        "('duped_column_1', 'duped_column_2')). Please create different transformer "
        'objects for each assignment.'
    )
    with pytest.raises(InvalidConfigError, match=expected_msg):
        _validate_unique_transformer_instances(column_name_to_transformer)


def test__validate_unique_transformer_instances_multiple_duplicates(column_name_to_transformer):
    """Test the function errors when many transformer instances are reused."""
    # Setup
    column_name_to_transformer = column_name_to_transformer.copy()
    column_name_to_transformer['duped_column_1'] = column_name_to_transformer['colA']
    column_name_to_transformer['duped_column_2'] = column_name_to_transformer['colA']
    column_name_to_transformer['duped_column_3'] = column_name_to_transformer['colB']
    column_name_to_transformer['duped_column_4'] = column_name_to_transformer['colB']

    # Run and Assert
    expected_msg = re.escape(
        "The same transformer instances are being assigned to columns ('colA', 'duped_column_1', "
        "'duped_column_2'), columns ('colB', 'duped_column_3', 'duped_column_4'). Please create "
        'different transformer objects for each assignment.'
    )
    with pytest.raises(InvalidConfigError, match=expected_msg):
        _validate_unique_transformer_instances(column_name_to_transformer)
