import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from rdt import HyperTransformer
from rdt.transformers import NumericalTransformer, get_transformers_by_type
from tests.quality.utils import download_single_table

R2_THRESHOLD = 0.2
TEST_THRESHOLD = 0.35
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


def get_transformer_regression_scores(data, data_type, dataset_name, transformers, metadata=None):
    """Returns regression scores for a list of transformers.

    Args:
        data (pandas.DataFrame):
            The dataset containing columns to predict and train with.
        data_type (string):
            The data type of the transformer.
        dataset_name (string):
            The name of the dataset.
        transformers (list):
            List of transformer instances.
        metadata (dict):
            Dictionary containing metadata for the table.

    Returns:
        pandas.DataFrame containing the score for each column predicted
        in the dataset. To get the scores, a regression model is trained.
        The features used are the output of transforming all the columns
        of the data type using a transformer in the transformers list.
    """
    columns_to_predict = find_columns(data, 'numerical')
    columns_to_transform = find_columns(data, data_type, metadata)
    scores = pd.DataFrame()
    features = data[columns_to_transform]

    for column in columns_to_predict:
        target = data[column].to_frame()
        numerical_transformer = NumericalTransformer(null_column=False)
        target = numerical_transformer.fit_transform(target, list(target.columns))
        target = format_array(target)
        for transformer in transformers:
            ht = HyperTransformer(default_data_type_transformers={data_type: transformer})
            ht.fit(features)
            transformed_features = ht.transform(features).to_numpy()
            score = get_regression_score(transformed_features, target)
            row = pd.Series({
                'transformer_name': transformer.__name__,
                'dataset_name': dataset_name,
                'column': column,
                'score': score
            })
            scores = scores.append(row, ignore_index=True)

    return scores


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


def get_regression_scores(test_cases, transformers_by_type):
    """Create table of all regression scores for test cases.

    Args:
        test_cases (list):
            List of test cases. Each test case is a tuple containing
            the dataset name, the name of the table to use from the
            dataset, and the data types to test against for that table.
        transformers_by_type (dict):
            Dict mapping data type to list of transformers that have that
            type as their input type.

    Returns:
        DataFrame where each row has a dataset name, transformer name,
        column name and regression score. The regression score is the
        coefficient of determination for the transformer predicting the column.
    """
    all_scores = defaultdict(pd.DataFrame)
    for dataset_name, table_name, data_types in test_cases:
        (data, metadata) = download_single_table(dataset_name, table_name)
        for data_type in data_types:
            transformers = transformers_by_type[data_type]
            regression_scores = get_transformer_regression_scores(
                data, data_type, dataset_name, transformers, metadata)
            all_scores[data_type] = all_scores[data_type].append(
                regression_scores, ignore_index=True)

    return all_scores


def get_results_table(regression_scores):
    """Create a table of results for each transformer on each dataset.

    Args:
        regression_scores (dict):
            Dict mapping data types to a DataFrame where each row has
            a table name, column name, transformer name and coefficient
            of determination for that transformer predicting that column.

    Returns:
        A DataFrame where each row has a transformer name, dataset name,
        average score for the dataset and a score comparing the transformer's
        average score for the dataset to the average of the average score for
        the dataset across all transformers of the same data type.
    """
    results = pd.DataFrame()
    for _, scores in regression_scores.items():
        table_column_groups = scores.groupby(['dataset_name', 'column'])
        valid = []
        for _, frame in table_column_groups:
            if frame['score'].mean() >= R2_THRESHOLD:
                valid.extend(frame.index)

        valid_scores = scores.loc[valid]
        transformer_dataset_groups = valid_scores.groupby(['dataset_name', 'transformer_name'])
        for (dataset_name, transformer_name), frame in transformer_dataset_groups:
            transformer_average = frame['score'].mean()
            dataset_rows = (valid_scores['dataset_name'] == dataset_name)
            transformer_rows = (valid_scores['transformer_name'] != transformer_name)
            data_without_transformer = valid_scores.loc[
                dataset_rows & transformer_rows
            ]
            average_without_transformer = data_without_transformer['score'].mean()

            row = pd.Series({
                'transformer_name': transformer_name,
                'dataset_name': dataset_name,
                'score': transformer_average,
                'score_relative_to_average': transformer_average / average_without_transformer
            })
            results = results.append(row, ignore_index=True)

    return results


def test_quality(subtests):
    """Run all the quality test cases.

    This test has multiple steps.
        1. It creates a list of test cases. Each test case has a dataset
        and a set of data types to test for the dataset.
        2. A dictionary is created mapping data types to a DataFrame
        containing the regression scores obtained from running the
        transformers of that type against the datasets in the test cases.
        Each row in the DataFrame has the transformer name, dataset name,
        column name and score. The scores are computed as follows:
            - For every transformer of the data type, transform all the
            columns of that data type.
            - For every numerical column in the dataset, the transformed
            columns are used as features to train a regression model.
            - The score is the coefficient of determination obtained from
            that model trying to predict the target column.
        3. Once the scores are gathered, a results table is created. Each row has
        a transformer name, dataset name, average score for the dataset and a score
        comparing the transformer's average score for the dataset to the average
        of the average score for the dataset across all transformers of the same
        data type.
        4. For every unique transformer in the results, a test is run to check
        that the transformer's score for each table is either higher than the
        threshold, or the comparitive score is higher than the threshold.
    """
    transformers_by_type = get_transformers_by_type()
    data_types_to_test = {
        data_type
        for data_type in transformers_by_type.keys()
        if data_type not in TYPES_TO_SKIP
    }
    test_cases = get_test_cases(data_types_to_test)
    all_regression_scores = get_regression_scores(test_cases, transformers_by_type)
    results = get_results_table(all_regression_scores)

    for transformer, frame in results.groupby('transformer_name'):
        with subtests.test(
                msg=f'Testing transformer {transformer}',
                transformer=transformer):
            relative_scores = frame['score_relative_to_average']
            assert all((relative_scores > TEST_THRESHOLD) | (frame['score'] > TEST_THRESHOLD))
