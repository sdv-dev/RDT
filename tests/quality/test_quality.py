import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from rdt import HyperTransformer
from rdt.transformers import NumericalTransformer, get_transformers_by_type
from tests.quality.utils import download_single_table_dataset

THRESHOLD = 0.4

TYPE_TO_DTYPE = {
    'numerical': ['number'],
    'float': ['float'],
    'int': ['int'],
    'categorical': ['object', 'category'],
    'datetime': ['datetime'],
}


def format_array(array):
    if not isinstance(array, np.ndarray):
        array = array.to_numpy()

    if len(array.shape) == 1:
        array = array.reshape(-1, 1)
    return array


def get_regression_score(features, target):
    model = LinearRegression()
    scores = cross_val_score(model, features, target)
    return np.mean(scores)


def find_columns(data, data_type):
    dtypes = TYPE_TO_DTYPE.get(data_type, data_type)
    columns = set()
    for dtype in dtypes:
        selected = data.select_dtypes(dtype)
        columns.update(set(selected.columns))

    return columns


def get_transformer_scores(data, data_type, transformers):
    columns_to_predict = find_columns(data, 'numerical')
    columns_to_transform = find_columns(data, data_type)
    all_scores = pd.DataFrame()
    index = [transformer.__name__ for transformer in transformers]
    features = data[columns_to_transform]

    for column in columns_to_predict:
        target = data[column].to_frame()
        numerical_transformer = NumericalTransformer(null_column=False)
        target = numerical_transformer.fit_transform(target, list(target.columns))
        target = format_array(target)
        scores = []
        for transformer in transformers:
            ht = HyperTransformer(data_type_transformers={data_type: transformer})
            ht.fit(features)
            transformed_features = ht.transform(features).to_numpy()
            scores.append(get_regression_score(transformed_features, target))
        all_scores[column] = pd.Series(scores, index=index)
    return all_scores


def validate_relative_score(scores, transformer):
    scores_without_transformer = scores.drop(transformer)
    means = scores_without_transformer.mean()
    means_above_threshold = means > THRESHOLD
    standard_deviations = scores_without_transformer.std()
    minimum_scores = means[means_above_threshold] - standard_deviations[means_above_threshold]

    assert all(scores.loc[transformer, means_above_threshold] > minimum_scores)


def get_test_cases():
    max_size = 5000000
    test_cases = []
    types_to_skip = {'numerical', 'float', 'integer'}
    path = os.path.join(os.path.dirname(__file__), 'dataset_info.csv')
    datasets = pd.read_csv(path)
    for _, row in datasets.iterrows():
        if row['modality'] == 'single-table' and row['table_size'] < max_size:
            for data_type in eval(row['table_types']):
                if data_type not in types_to_skip:
                    test_cases.append((data_type, row['name']))

    return test_cases


test_cases = get_test_cases()


@pytest.mark.parametrize('test_case', test_cases)
def test_quality(subtests, test_case):
    """Run all the quality test cases.

    This test goes through each test case and tests all the transformers
    of the test case's type against the test case's dataset. First, all
    the transformers of the data type are used to transform every column
    in the dataset of that type. A regression model is then trained on
    those transformed column to try and predict every numerical column
    in the dataset. The scores for each transformer are then compared to
    the mean score of the rest of the transformers to make sure none are more
    than one standard deviation away.
    """
    dataset_name = test_case[1]
    data = download_single_table_dataset(dataset_name)
    data_type = test_case[0]
    transformers = get_transformers_by_type()[data_type]
    scores = get_transformer_scores(data, data_type, transformers)

    for transformer in scores.index:
        with subtests.test(
                msg=f'Testing transformer {transformer} with dataset {dataset_name}',
                transformer=transformer):
            validate_relative_score(scores, transformer)
