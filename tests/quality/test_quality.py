import os.path as op

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from rdt import HyperTransformer
from rdt.transformers import NumericalTransformer, get_transformers_by_type

TEST_PREFIX = op.join(op.dirname(__file__), 'datasets')
TEST_CASES = [
    ('categorical', 'adult.csv')
]
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


def get_regression_score(features, target, training_size_perc=0.8):
    training_size = round(training_size_perc * features.shape[0])
    y_training = target[0:training_size]
    X_training = features[0:training_size]
    model = LinearRegression().fit(X_training, y_training)

    y_test = target[training_size:]
    X_test = features[training_size:]
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions)


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
    standard_deviations = scores_without_transformer.std()
    for column in means.index:
        mean_score = means[column]
        std = standard_deviations[column]
        if mean_score > THRESHOLD:
            assert scores.loc[transformer, column] > mean_score - std


@pytest.mark.parametrize('test_case', TEST_CASES)
def test_quality(test_case):
    """Run all the quality test cases.
    """
    data_type = test_case[0]
    file_name = op.join(TEST_PREFIX, test_case[1])
    data = pd.read_csv(file_name)
    transformers = get_transformers_by_type()[data_type]
    scores = get_transformer_scores(data, data_type, transformers)
    for transformer in scores.index:
        validate_relative_score(scores, transformer)
