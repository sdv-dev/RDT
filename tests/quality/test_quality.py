import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from rdt import HyperTransformer
from rdt.transformers import NumericalTransformer, get_transformers_by_type
from tests.quality.utils import download_single_table

THRESHOLD = 0.2
MAX_SIZE = 5000000
TYPES_TO_SKIP = {'numerical', 'float', 'integer', 'id', None}

TYPE_TO_DTYPE = {
    'numerical': ['number'],
    'float': ['float'],
    'int': ['int'],
    'categorical': ['object', 'category'],
    'datetime': ['datetime'],
    'boolean': ['bool']
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


def find_columns(data, data_type, metadata=None):
    if metadata:
        return {
            column
            for column in metadata['fields']
            if metadata['fields'][column]['type'] == data_type
        }

    columns = set()
    dtypes = TYPE_TO_DTYPE.get(data_type, data_type)
    for dtype in dtypes:
        selected = data.select_dtypes(dtype)
        columns.update(set(selected.columns))

    return columns


def get_transformer_scores(data, data_type, transformers, metadata=None):
    columns_to_predict = find_columns(data, 'numerical')
    columns_to_transform = find_columns(data, data_type, metadata)
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
    if len(scores_without_transformer) == 0:
        return

    means = scores_without_transformer.mean()
    means_above_threshold = means > THRESHOLD
    standard_deviations = scores.std()
    minimum_scores = means[means_above_threshold] - 2 * standard_deviations[means_above_threshold]

    assert all(scores.loc[transformer, means_above_threshold] >= minimum_scores)


def get_test_cases():
    test_cases = []
    path = os.path.join(os.path.dirname(__file__), 'dataset_info.csv')
    datasets = pd.read_csv(path)
    for _, row in datasets.iterrows():
        if row['table_size'] < MAX_SIZE and row['modality'] in ['single-table', 'timeseries']:
            for data_type in eval(row['table_types']):
                if data_type not in TYPES_TO_SKIP:
                    test_cases.append((data_type, row['name'], row['table_name']))

    return test_cases


def test_quality(subtests):
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
    transformers_by_type = get_transformers_by_type()
    test_cases = get_test_cases()
    data_types_to_test = [
        data_type
        for data_type in transformers_by_type.keys()
        if data_type not in TYPES_TO_SKIP
    ]
    tested_data_types = dict.fromkeys(data_types_to_test, False)

    for data_type, dataset_name, table_name in test_cases:
        transformers = transformers_by_type[data_type]
        (data, metadata) = download_single_table(dataset_name, table_name)
        scores = get_transformer_scores(data, data_type, transformers, metadata)
        if any(scores.mean() > THRESHOLD):
            tested_data_types[data_type] = True

        for transformer in scores.index:
            with subtests.test(
                    msg=f'Testing transformer {transformer} with dataset {dataset_name}',
                    transformer=transformer):
                validate_relative_score(scores, transformer)

    assert all(tested_data_types[data_type] is True for data_type in tested_data_types)
