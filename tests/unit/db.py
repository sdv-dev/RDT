import pandas as pd

from rdt.hyper_transformer import HyperTransformer
from rdt.transformers.base import BaseTransformer
from rdt.transformers.categorical import FrequencyEncoder
from rdt.transformers.numerical import FloatFormatter

from unittest.mock import Mock, call, patch

def get_transformed_data(drop=False):
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': ['a', 'a', 'b', 'a'],
        'bool': [False, False, True, False],
        'datetime': pd.to_datetime(['2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01']),
        'integer.out': ['1', '2', '1', '3'],
        'integer.out': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': [0.375, 0.375, 0.875, 0.375],
        'bool': [0.0, 0.0, 1.0, 0.0],
        'datetime': [
            1.2649824e+18,
            1.262304e+18,
            1.2649824e+18,
            1.262304e+18
        ]
    })

    if drop:
        return data.drop([
            'integer',
            'float',
            'categorical',
            'bool',
            'datetime',
            'integer.out'
        ], axis=1)

    return data

def get_data():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': ['a', 'a', 'b', 'a'],
        'bool': [False, False, True, False],
        'datetime': pd.to_datetime(['2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'])
    })

int_transformer = Mock()
float_transformer = Mock()
generator_transformer = Mock()
int_transformer.get_output_columns.return_value = ['integer.out']
float_transformer.get_output_columns.return_value = ['float']
generator_transformer.get_output_columns.return_value = []

reverse_transformed_data = get_transformed_data()
float_transformer.reverse_transform = lambda x: x
int_transformer.reverse_transform.return_value = reverse_transformed_data

data = get_transformed_data(True)

ht = HyperTransformer()
ht._validate_config_exists = Mock()
ht._validate_config_exists.return_value = True
ht._fitted = True
ht._transformers_sequence = [
    int_transformer,
    float_transformer,
    generator_transformer
]
ht._output_columns = list(data.columns)
expected = get_data()
ht._input_columns = list(expected.columns)

# Run
reverse_transformed = ht.reverse_transform_subset(data)

# Assert
print(reverse_transformed)

