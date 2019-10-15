from unittest.mock import patch

import numpy as np
import pandas as pd

from rdt import HyperTransformer


def get_input_data():
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1],
        'categorical': ['a', 'b', np.nan, 'b', 'a'],
        'bool': [False, np.nan, False, True, False],
        'datetime': [
            np.nan, '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
        ],
        'names': ['Jon', 'Arya', 'Sansa', 'Jon', 'Robb'],
    })
    data['datetime'] = pd.to_datetime(data['datetime'])

    return data


def get_transformed_data():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.125, 0.1],
        'float#1': [0.0, 0.0, 0.0, 1.0, 0.0],
        'categorical': [0.6, 0.2, 0.9, 0.2, 0.6],
        'bool': [0.0, -1.0, 0.0, 1.0, 0.0],
        'bool#1': [0.0, 1.0, 0.0, 0.0, 0.0],
        'datetime': [
            1.2636432e+18, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime#1': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.2, 0.9, 0.5, 0.2, 0.7],
    })


def get_transformers():
    return {
        'integer': {
            'class': 'NumericalTransformer',
            'kwargs': {
                'dtype': int,
            }
        },
        'float': {
            'class': 'NumericalTransformer',
            'kwargs': {
                'dtype': float,
            }
        },
        'categorical': {
            'class': 'CategoricalTransformer'
        },
        'bool': {
            'class': 'BooleanTransformer'
        },
        'datetime': {
            'class': 'DatetimeTransformer'
        },
        'names': {
            'class': 'CategoricalTransformer',
            'kwargs': {
                'anonymize': 'first_name'
            }
        },
    }


@patch('rdt.transformers.categorical.Faker')
def test_hypertransformer_with_transformers(faker_mock):
    faker_mock.return_value.first_name.side_effect = ['Jaime', 'Cersei', 'Tywin', 'Tyrion']
    data = get_input_data()
    transformers = get_transformers()

    ht = HyperTransformer(transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).values,
        expected.sort_index(axis=1).values
    )

    reversed_data = ht.reverse_transform(transformed)

    original_names = data.pop('names')
    reversed_names = reversed_data.pop('names')

    pd.testing.assert_frame_equal(data.sort_index(axis=1), reversed_data.sort_index(axis=1))

    for name in original_names:
        assert name not in reversed_names


@patch('rdt.transformers.categorical.Faker')
def test_hypertransformer_without_transformers(faker_mock):
    faker_mock.return_value.first_name.side_effect = ['Jaime', 'Cersei', 'Tywin', 'Tyrion']
    data = get_input_data()

    ht = HyperTransformer()
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).values,
        expected.sort_index(axis=1).values
    )

    reversed_data = ht.reverse_transform(transformed)

    original_names = data.pop('names')
    reversed_names = reversed_data.pop('names')

    pd.testing.assert_frame_equal(data.sort_index(axis=1), reversed_data.sort_index(axis=1))

    for name in original_names:
        assert name not in reversed_names
