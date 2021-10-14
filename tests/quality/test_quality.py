import os
from collections import defaultdict

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


def get_dataset_transformer_scores(data, data_type, transformers, metadata=None):
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


def get_test_cases(data_types):
    test_cases = []
    path = os.path.join(os.path.dirname(__file__), 'dataset_info.csv')
    datasets = pd.read_csv(path)
    for _, row in datasets.iterrows():
        if row['table_size'] < MAX_SIZE and row['modality'] == 'single-table':
            table_types = eval(row['table_types'])
            table_types_to_test = data_types.intersection(table_types)
            if len(table_types_to_test) > 0:
                test_cases.append((row['name'], row['table_name'], table_types_to_test))

    return test_cases


def test_quality(subtests):
    """Run all the quality test cases.

    This test has multiple steps.
        1. It creates a list of test cases. Each test case has a dataset
        and a set of data types to test for the dataset.
        2. A dictionary is created mapping data types to another dict mapping
        datasets to the scores for that dataset with that data type. This is
        done by looping through the test cases and doing the following:
            - For every transformer of the data type, transform all the
            columns of that data type.
            - For every numerical column in the dataset, the transformed
            columns are used as features to train a regression model.
            - A DataFrame of scores is created where the index is the
            transformer name, and the column values are the scores for
            the different numerical columns in the dataset.
        3. Once the scores are gathered, the transformers for each data type
        are looped through and the following happens:
            - The dict mapping datasets to scores is pulled for the type.
            - If it is empty, this means no datasets had high enough scores
            and the test fails.
            - Otherwise, for each DataFrame of scores, the transformer in
            question's score is compared to the mean score of the other
            transformers. If it is within two standard deviations, the test passes.
    """
    transformers_by_type = get_transformers_by_type()
    data_types_to_test = {
        data_type
        for data_type in transformers_by_type.keys()
        if data_type not in TYPES_TO_SKIP
    }
    test_cases = get_test_cases(data_types_to_test)

    all_scores = defaultdict(dict)
    for dataset_name, table_name, data_types in test_cases:
        (data, metadata) = download_single_table(dataset_name, table_name)
        for data_type in data_types:
            transformers = transformers_by_type[data_type]
            scores = get_dataset_transformer_scores(data, data_type, transformers, metadata)
            if any(scores.mean() > THRESHOLD):
                all_scores[data_type][dataset_name] = scores

    for data_type in all_scores:
        for transformer in transformers_by_type[data_type]:
            transformer_name = transformer.__name__
            data_type_results = all_scores[data_type]
            with subtests.test(
                    msg=f'Testing transformer {transformer_name}',
                    transformer=transformer):
                assert data_type_results
                for dataset in data_type_results:
                    scores = data_type_results[dataset]
                    validate_relative_score(scores, transformer_name)
