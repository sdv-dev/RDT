import contextlib
import io
import re
from collections import defaultdict
from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.errors import Error, NotFittedError
from rdt.transformers import (
    BinaryEncoder, FloatFormatter, FrequencyEncoder, GaussianNormalizer, LabelEncoder,
    OneHotEncoder, UnixTimestampEncoder)


class TestHyperTransformer(TestCase):

    @patch('rdt.hyper_transformer.print')
    def test__user_message_no_prefix(self, mock_print):
        """Test that the ``_user_message`` function prints a message.

        Mock:
            - Mock print function.

        Side Effects:
            - Print has been called once with message.
        """
        # Run
        HyperTransformer._user_message('My message.')

        # Assert
        mock_print.assert_called_once_with('My message.')

    @patch('rdt.hyper_transformer.print')
    def test__user_message_with_prefix(self, mock_print):
        """Test that the ``_user_message`` function prints a message with prefix.

        Mock:
            - Mock print function.

        Side Effects:
            - Print has been called once with the prefix followed by the message.
        """
        # Run
        HyperTransformer._user_message('My tip.', 'Tip')

        # Assert
        mock_print.assert_called_once_with('Tip: My tip.')

    def test__add_field_to_set_string(self):
        """Test the ``_add_field_to_set`` method.

        Test that ``field`` is added to the ``field_set``.

        Input:
            - a field name.
            - a set of field names.

        Expected behavior:
            - the passed field name should be added to the set of field names.
        """
        # Setup
        ht = HyperTransformer()
        field = 'abc'
        field_set = {'def', 'ghi'}

        # Run
        ht._add_field_to_set(field, field_set)

        # Assert
        assert field_set == {'abc', 'def', 'ghi'}

    def test__add_field_to_set_tuple(self):
        """Test the ``_add_field_to_set`` method when given a tuple.

        Test that each ``field`` name is added to the ``field_set``.

        Input:
            - a tuple of field names.
            - a set of field names.

        Expected behavior:
            - the passed field names should be added to the set of field names.
        """
        # Setup
        ht = HyperTransformer()
        field = ('abc', 'jkl')
        field_set = {'def', 'ghi'}

        # Run
        ht._add_field_to_set(field, field_set)

        # Assert
        assert field_set == {'abc', 'def', 'ghi', 'jkl'}

    def test__validate_field_transformers(self):
        """Test the ``_validate_field_transformers`` method.

        Tests that a ``ValueError`` is raised if a field has multiple
        definitions in the ``field_transformers`` dict.

        Setup:
            - field_transformers set to have duplicate definitions.

        Expected behavior:
            - A ``ValueError`` should be raised if a field is defined
            more than once.
        """
        # Setup
        int_transformer = Mock()
        float_transformer = Mock()

        field_transformers = {
            'integer': int_transformer,
            'float': float_transformer,
            ('integer',): int_transformer
        }
        ht = HyperTransformer()
        ht.field_transformers = field_transformers

        # Run / Assert
        error_msg = (
            r'Multiple transformers specified for the field \(\'integer\',\). '
            'Each field can have at most one transformer defined in field_transformers.'
        )

        with pytest.raises(ValueError, match=error_msg):
            ht._validate_field_transformers()

    @patch('rdt.hyper_transformer.HyperTransformer._create_multi_column_fields')
    @patch('rdt.hyper_transformer.HyperTransformer._validate_field_transformers')
    def test___init__(self, validation_mock, multi_column_mock):
        """Test create new instance of HyperTransformer"""
        # Run
        ht = HyperTransformer()

        # Asserts
        assert ht.field_sdtypes == {}
        assert ht._default_sdtype_transformers == {}
        assert ht.field_transformers == {}
        assert ht._specified_fields == set()
        assert ht._valid_output_sdtypes == ht._DEFAULT_OUTPUT_SDTYPES
        assert ht._transformers_sequence == []
        assert ht._output_columns == []
        assert ht._input_columns == []
        assert ht._fitted_fields == set()
        assert ht._fitted is False
        assert ht._modified_config is False
        assert ht._transformers_tree == defaultdict(dict)
        multi_column_mock.assert_called_once()
        validation_mock.assert_called_once()

    def test__unfit(self):
        """Test the ``_unfit`` method.

        The ``_unfit`` method should reset most attributes of the HyperTransformer.

        Setup:
            - instance._fitted is set to True
            - instance._transformers_sequence is a list of transformers
            - instance._output_columns is a list of columns
            - instance._input_columns is a list of columns

        Expected behavior:
            - instance._fitted is set to False
            - instance._transformers_sequence is set to []
            - instance._output_columns is an empty list
            - instance._input_columns is an empty list
        """
        # Setup
        ht = HyperTransformer()
        ht._fitted = True
        ht._transformers_sequence = [BinaryEncoder(), FloatFormatter()]
        ht._output_columns = ['col1', 'col2']
        ht._input_columns = ['col3', 'col4']

        # Run
        ht._unfit()

        # Assert
        assert ht._fitted is False
        assert ht._transformers_sequence == []
        assert ht._fitted_fields == set()
        assert ht._output_columns == []
        assert ht._input_columns == []
        assert ht._transformers_tree == {}

    def test__create_multi_column_fields(self):
        """Test the ``_create_multi_column_fields`` method.

        This tests that the method goes through both the ``field_transformers``
        dict and ``field_sdtypes`` dict to find multi_column fields and map
        each column to its corresponding tuple.

        Setup:
            - instance.field_transformers will be populated with multi-column fields
            - instance.field_sdtypes will be populated with multi-column fields

        Output:
            - A dict mapping each column name that is part of a multi-column
            field to the tuple of columns in the field it belongs to.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'a': BinaryEncoder,
            'b': UnixTimestampEncoder,
            ('c', 'd'): UnixTimestampEncoder,
            'e': FloatFormatter
        }
        ht.field_sdtypes = {
            'f': 'categorical',
            ('g', 'h'): 'datetime'
        }

        # Run
        multi_column_fields = ht._create_multi_column_fields()

        # Assert
        expected = {
            'c': ('c', 'd'),
            'd': ('c', 'd'),
            'g': ('g', 'h'),
            'h': ('g', 'h')
        }
        assert multi_column_fields == expected

    def test__get_next_transformer_field_transformer(self):
        """Test the ``_get_next_transformer method.

        This tests that if the transformer is defined in the
        ``instance.field_transformers`` dict, then it is returned
        even if the output sdtype is final.

        Setup:
            - field_transformers is given a transformer for the
            output field.
            - _default_sdtype_transformers will be given a different transformer
            for the output sdtype of the output field.

        Input:
            - An output field name in field_transformers.
            - Output sdtype  is numerical.
            - next_transformers is None.

        Output:
            - The transformer defined in field_transformers.
        """
        # Setup
        transformer = FloatFormatter()
        ht = HyperTransformer()
        ht.field_transformers = {'a.out': transformer}
        ht._default_sdtype_transformers = {'numerical': GaussianNormalizer()}

        # Run
        next_transformer = ht._get_next_transformer('a.out', 'numerical', None)

        # Assert
        assert next_transformer == transformer

    def test__get_next_transformer_final_output_sdtype(self):
        """Test the ``_get_next_transformer method.

        This tests that if the transformer is not defined in the
        ``instance.field_transformers`` dict and its output sdtype
        is in ``instance._transform_output_sdtypes``, then ``None``
        is returned.

        Setup:
            - _default_sdtype_transformers will be given a transformer
            for the output sdtype of the output field.

        Input:
            - An output field name in field_transformers.
            - Output sdtype  is numerical.
            - next_transformers is None.

        Output:
            - None.
        """
        # Setup
        ht = HyperTransformer()
        ht._default_sdtype_transformers = {'numerical': GaussianNormalizer()}

        # Run
        next_transformer = ht._get_next_transformer('a.out', 'numerical', None)

        # Assert
        assert next_transformer is None

    def test__get_next_transformer_next_transformers(self):
        """Test the ``_get_next_transformer method.

        This tests that if the transformer is not defined in the
        ``instance.field_transformers`` dict and its output sdtype
        is not in ``instance._transform_output_sdtypes`` and the
        ``next_transformers`` dict has a transformer for the output
        field, then it is used.

        Setup:
            - _default_sdtype_transformers will be given a transformer
            for the output sdtype of the output field.

        Input:
            - An output field name in field_transformers.
            - Output sdtype  is categorical.
            - next_transformers is has a transformer defined
            for the output field.

        Output:
            - The transformer defined in next_transformers.
        """
        # Setup
        transformer = FrequencyEncoder()
        ht = HyperTransformer()
        ht._default_sdtype_transformers = {'categorical': OneHotEncoder()}
        next_transformers = {'a.out': transformer}

        # Run
        next_transformer = ht._get_next_transformer('a.out', 'categorical', next_transformers)

        # Assert
        assert next_transformer == transformer

    @patch('rdt.transformers.get_default_transformer')
    def test__get_next_transformer_default_transformer(self, mock):
        """Test the ``_get_next_transformer method.

        This tests that if the transformer is not defined in the
        ``instance.field_transformers`` dict or ``next_transformers``
        and its output sdtype is not in ``instance._transform_output_sdtypes``
        then the default_transformer is used.

        Setup:
            - A mock is used for ``get_default_transformer``.

        Input:
            - An output field name in field_transformers.
            - Output sdtype  is categorical.
            - next_transformers is None.

        Output:
            - The default transformer.
        """
        # Setup
        transformer = FrequencyEncoder(add_noise=True)
        mock.return_value = transformer
        ht = HyperTransformer()
        ht._default_sdtype_transformers = {'categorical': OneHotEncoder()}

        # Run
        next_transformer = ht._get_next_transformer('a.out', 'categorical', None)

        # Assert
        assert isinstance(next_transformer, FrequencyEncoder)
        assert next_transformer.add_noise is False

    @patch('rdt.hyper_transformer.get_default_transformer')
    def test__learn_config(self, get_default_transformer_mock):
        """Test the ``_learn_config_method.

        Tests that the method learns the ``sdtype`` and ``transformer`` for every field
        that doesn't already have one. If it doesn't find a transformer it should use
        ``_default_sdtype_transformers`` and if that also doesn't have one it should use
        ``get_default_transformer``.

        Setup:
            - A mock for ``get_default_tranformer``.
            - ``field_transformers`` partially provided.
            - ``field_sdtypes`` partially provided.

        Input:
            - A DataFrame with multiple columns of different sdtypes.

        Side effects:
            - The appropriate ``sdtypes`` and ``transformers`` should be found.
        """
        # Setup
        int_transformer = Mock()
        float_transformer = Mock()
        categorical_transformer = Mock()
        bool_transformer = Mock()
        datetime_transformer = Mock()

        data = self.get_data()
        field_transformers = {
            'integer': int_transformer,
            'float': float_transformer,
        }
        default_sdtype_transformers = {
            'boolean': bool_transformer,
            'categorical': categorical_transformer
        }
        get_default_transformer_mock.return_value = datetime_transformer
        ht = HyperTransformer()
        ht.field_transformers = field_transformers
        ht.field_sdtypes = {'datetime': 'datetime'}
        ht._default_sdtype_transformers = default_sdtype_transformers
        ht._unfit = Mock()

        # Run
        ht._learn_config(data)

        # Assert
        assert ht.field_sdtypes == {
            'integer': 'numerical',
            'float': 'numerical',
            'bool': 'boolean',
            'categorical': 'categorical',
            'datetime': 'datetime'
        }
        assert ht.field_transformers == {
            'integer': int_transformer,
            'float': float_transformer,
            'categorical': categorical_transformer,
            'bool': bool_transformer,
            'datetime': datetime_transformer
        }
        ht._unfit.assert_called_once()

    def test_detect_initial_config(self):
        """Test the ``detect_initial_config`` method.

        This tests that ``field_sdtypes`` and ``field_transformers`` are correctly set,
        ``_config_detected`` is set to True, and that the appropriate configuration is printed.

        Input:
            - A DataFrame.
        """
        # Setup
        ht = HyperTransformer()
        data = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': ['a', 'b', 'c'],
            'col3': [True, False, True],
            'col4': pd.to_datetime(['2010-02-01', '2010-01-01', '2010-02-01']),
            'col5': [1, 2, 3]
        })

        # Run
        f_out = io.StringIO()
        with contextlib.redirect_stdout(f_out):
            ht.detect_initial_config(data)

        output = f_out.getvalue()

        # Assert
        assert ht.field_sdtypes == {
            'col1': 'numerical',
            'col2': 'categorical',
            'col3': 'boolean',
            'col4': 'datetime',
            'col5': 'numerical'
        }

        field_transformers = {k: repr(v) for (k, v) in ht.field_transformers.items()}
        assert field_transformers == {
            'col1': "FloatFormatter(missing_value_replacement='mean')",
            'col2': 'FrequencyEncoder()',
            'col3': "BinaryEncoder(missing_value_replacement='mode')",
            'col4': "UnixTimestampEncoder(missing_value_replacement='mean')",
            'col5': "FloatFormatter(missing_value_replacement='mean')"
        }

        expected_output = '\n'.join((
            'Detecting a new config from the data ... SUCCESS',
            'Setting the new config ... SUCCESS',
            'Config:',
            '{',
            '    "sdtypes": {',
            '        "col1": "numerical",',
            '        "col2": "categorical",',
            '        "col3": "boolean",',
            '        "col4": "datetime",',
            '        "col5": "numerical"',
            '    },',
            '    "transformers": {',
            '        "col1": FloatFormatter(missing_value_replacement=\'mean\'),',
            '        "col2": FrequencyEncoder(),',
            '        "col3": BinaryEncoder(missing_value_replacement=\'mode\'),',
            '        "col4": UnixTimestampEncoder(missing_value_replacement=\'mean\'),',
            '        "col5": FloatFormatter(missing_value_replacement=\'mean\')',
            '    }',
            '}',
            ''
        ))
        assert output == expected_output

    @patch('rdt.hyper_transformer.get_transformer_instance')
    def test__fit_field_transformer(self, get_transformer_instance_mock):
        """Test the ``_fit_field_transformer`` method.

        This tests that the ``_fit_field_transformer`` behaves as expected.
        It should fit the transformer it is provided, loops through its
        outputs, transform the data if the output is not ML ready and call
        itself recursively if it can.

        Setup:
            - A mock for ``get_transformer_instance``.
            - A mock for the transformer returned by ``get_transformer_instance``.
            The ``get_output_sdtypes`` method will return two outputs, one that
            is ML ready and one that isn't.

        Input:
            - A DataFrame with one column.
            - A column name to fit the transformer to.
            - A transformer.

        Output:
            - A DataFrame with columns that result from transforming the
            outputs of the original transformer.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        transformed_data1 = pd.DataFrame({
            'a.out1': ['2', '4', '6'],
            'a.out2': [1, 2, 3]
        })
        transformer1 = Mock()
        transformer2 = Mock()
        transformer1.get_output_sdtypes.return_value = {
            'a.out1': 'categorical',
            'a.out2': 'numerical'
        }
        transformer1.get_next_transformers.return_value = {
            'a.out1': transformer2
        }
        transformer1.transform.return_value = transformed_data1
        transformer2.get_output_sdtypes.return_value = {
            'a.out1.value': 'numerical'
        }
        transformer2.get_next_transformers.return_value = None
        get_transformer_instance_mock.side_effect = [
            transformer1,
            transformer2,
            None,
            None
        ]
        ht = HyperTransformer()
        ht._get_next_transformer = Mock()
        ht._get_next_transformer.side_effect = [
            transformer2,
            None,
            None
        ]

        # Run
        out = ht._fit_field_transformer(data, 'a', transformer1)

        # Assert
        expected = pd.DataFrame({
            'a.out1': ['2', '4', '6'],
            'a.out2': [1, 2, 3]
        })
        pd.testing.assert_frame_equal(out, expected)
        transformer1.fit.assert_called_once()
        transformer1.transform.assert_called_once_with(data)
        transformer2.fit.assert_called_once()
        assert ht._transformers_sequence == [transformer1, transformer2]
        assert ht._transformers_tree == {
            'a': {'transformer': transformer1, 'outputs': ['a.out1', 'a.out2']},
            'a.out1': {'transformer': transformer2, 'outputs': ['a.out1.value']}
        }

    @patch('rdt.hyper_transformer.get_transformer_instance')
    def test__fit_field_transformer_multi_column_field_not_ready(
        self,
        get_transformer_instance_mock
    ):
        """Test the ``_fit_field_transformer`` method.

        This tests that the ``_fit_field_transformer`` behaves as expected.
        If the column is part of a multi-column field, and the other columns
        aren't present in the data, then it should not fit the next transformer.
        It should however, transform the data.

        Setup:
            - A mock for ``get_transformer_instance``.
            - A mock for the transformer returned by ``get_transformer_instance``.
            The ``get_output_sdtypes`` method will return one output that is part of
            a multi-column field.
            - A mock for ``_multi_column_fields`` to return the multi-column field.

        Input:
            - A DataFrame with two columns.
            - A column name to fit the transformer to.
            - A transformer.

        Output:
            - A DataFrame with columns that result from transforming the
            outputs of the original transformer.
        """
        # Setup
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        transformed_data1 = pd.DataFrame({
            'a.out1': ['1', '2', '3'],
            'b': [4, 5, 6]
        })
        transformer1 = Mock()
        transformer2 = Mock()
        transformer1.get_output_sdtypes.return_value = {
            'a.out1': 'categorical'
        }
        transformer1.get_next_transformers.return_value = None
        transformer1.transform.return_value = transformed_data1
        get_transformer_instance_mock.side_effect = [transformer1]
        ht = HyperTransformer()
        ht._get_next_transformer = Mock()
        ht._get_next_transformer.side_effect = [transformer2]
        ht._multi_column_fields = Mock()
        ht._multi_column_fields.get.return_value = ('a.out1', 'b.out1')

        # Run
        out = ht._fit_field_transformer(data, 'a', transformer1)

        # Assert
        expected = pd.DataFrame({
            'a.out1': ['1', '2', '3'],
            'b': [4, 5, 6]
        })
        assert ht._output_columns == []
        pd.testing.assert_frame_equal(out, expected)
        transformer1.fit.assert_called_once()
        transformer1.transform.assert_called_once_with(data)
        transformer2.fit.assert_not_called()
        assert ht._transformers_sequence == [transformer1]

    def test__fit_field_transformer_transformer_is_none(self):
        """Test the ``_fit_field_transformer`` method.

        Test that when a ``transformer`` is ``None`` the ``outputs`` are the same
        as the field.

        Setup:
            - Dataframe with a column.
            - HyperTransformer instance.

        Input:
            - A DataFrame with two columns.
            - A column name to fit the transformer to.
            - A ``None`` value as ``transformer``.

        Output:
            - input data.

        Side Effects:
            - ``ht._transformers_sequence`` has not been updated.
            - ``ht._transformers_tree`` contains the ``column`` with a ``transformer`` as ``None``
              and ``outputs`` is the name of the column.
        """
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3]})
        ht = HyperTransformer()

        # Run
        out = ht._fit_field_transformer(data, 'a', None)

        # Assert
        pd.testing.assert_frame_equal(out, data)
        assert ht._transformers_sequence == []
        assert ht._transformers_tree == {
            'a': {'transformer': None, 'outputs': ['a']},
        }

    @patch('rdt.hyper_transformer.warnings')
    def test__validate_all_fields_fitted(self, warnings_mock):
        """Test the ``_validate_all_fields_fitted`` method.

        Tests that the ``_validate_all_fields_fitted`` method raises a warning
        if there are fields in ``field_transformers`` that were not fitted.

        Setup:
            - A mock for warnings.
            - A mock for ``field_transformers`` with a misspelled field.
            - A mock for ``_fitted_fields`` containing the other fields.

        Expected behavior:
            - Warnings should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._specified_fields = {'integer', 'float', 'categorical'}
        ht._fitted_fields = {'integer', 'float'}

        # Run
        ht._validate_all_fields_fitted()

        # Assert
        warnings_mock.warn.assert_called_once()

    def test__sort_output_columns(self):
        """Test the ``_sort_output_columns`` method.

        Assert the method correctly sorts the ``_output_columns`` attribute
        according to ``_input_columns``.

        Setup:
            - Initialize the ``HyperTransformer`` with:
                - A mock for the ``_get_final_output_columns`` method, with ``side_effect``
                set to the lists of generated output columns for each input column.
                - A list of columns names for ``_input_columns``.

        Expected behavior:
            - ``_output_columns`` should be sorted according to the ``_input_columns``.
        """
        # Setup
        ht = HyperTransformer()
        ht._get_final_output_columns = Mock()
        ht._get_final_output_columns.side_effect = [
            ['a.is_null'], ['b.value', 'b.is_null'], ['c.value']
        ]
        ht._input_columns = ['a', 'b', 'c']

        # Run
        ht._sort_output_columns()

        # Assert
        assert ht._output_columns == ['a.is_null', 'b.value', 'b.is_null', 'c.value']

    def test__validate_config(self):
        """Test the ``_validate_config`` method.

        The method should throw a warnings if the ``sdtypes`` of any column name doesn't match
        the ``sdtype`` of its transformer.

        Setup:
            - A mock for warnings.

        Input:
            - A config with a transformers dict that has a transformer that doesn't match the
            sdtype for the same column in sdtypes dict.

        Expected behavior:
            - There should be a warning.
        """
        # Setup
        transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        sdtypes = {
            'column1': 'numerical',
            'column2': 'numerical'
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers
        }

        # Run
        error_msg = re.escape(
            "Some transformers you've assigned are not compatible with the sdtypes. "
            "Please change the following columns: ['column2']"
        )
        with pytest.raises(Error, match=error_msg):
            HyperTransformer._validate_config(config)

    @patch('rdt.hyper_transformer.warnings')
    def test__validate_config_no_warning(self, warnings_mock):
        """Test the ``_validate_config`` method with no warning.

        The method should not throw a warnings if the ``sdtypes`` of all columns match
        the ``sdtype`` of their transformers.

        Setup:
            - A mock for warnings.

        Input:
            - A config with a transformers dict that matches the sdtypes for each column.

        Expected behavior:
            - There should be no warning.
        """
        # Setup
        transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        sdtypes = {
            'column1': 'numerical',
            'column2': 'categorical'
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers
        }

        # Run
        HyperTransformer._validate_config(config)

        # Assert
        warnings_mock.warn.assert_not_called()

    def test__validate_config_invalid_key(self):
        """Test the ``_validate_config`` method.

        The method should crash if an unexpected key is present in the config.

        Input:
            - A config with an unexpected key.

        Expected behavior:
            - It should raise an error.
        """
        # Setup
        transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        sdtypes = {
            'column1': 'numerical',
            'column2': 'numerical'
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers,
            'unexpected': 10
        }

        # Run / Assert
        error_msg = re.escape(
            'Error: Invalid config. Please provide 2 dictionaries '
            "named 'sdtypes' and 'transformers'."
        )
        with pytest.raises(Error, match=error_msg):
            HyperTransformer._validate_config(config)

    def test__validate_config_missing_sdtypes(self):
        """Test the ``_validate_config`` method.

        The method should crash if ``sdytpes`` is missing from the config.

        Input:
            - A config with only ``transformers``.

        Expected behavior:
            - It should raise an error.
        """
        # Setup
        transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        config = {
            'transformers': transformers,
        }

        # Run / Assert
        error_msg = re.escape(
            'Error: Invalid config. Please provide 2 dictionaries '
            "named 'sdtypes' and 'transformers'."
        )
        with pytest.raises(Error, match=error_msg):
            HyperTransformer._validate_config(config)

    def test__validate_config_mismatched_columns(self):
        """Test the ``_validate_config`` method.

        The method should crash if ``sdytpes`` and ``transformers`` have different of columns.

        Input:
            - A config with mismatched ``transformers`` and ``sdtypes`` .

        Expected behavior:
            - It should raise an error.
        """
        # Setup
        sdtypes = {
            'column1': 'numerical',
            'column2': 'numerical'
        }
        transformers = {
            'column1': FloatFormatter(),
            'column3': FrequencyEncoder()
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers,
        }

        # Run / Assert
        error_msg = re.escape(
            "The column names in the 'sdtypes' dictionary must match the "
            "column names in the 'transformers' dictionary."
        )
        with pytest.raises(Error, match=error_msg):
            HyperTransformer._validate_config(config)

    def test__validate_config_invalid_sdtype(self):
        """Test the ``_validate_config`` method.

        The method should crash if ``sdytpes`` is passed non-supported values.

        Input:
            - A config with incorrect ``sdytpes``.

        Expected behavior:
            - It should raise an error.
        """
        # Setup
        sdtypes = {
            'column1': 'numerical',
            'column2': 'unexpected'
        }
        transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers,
        }

        # Run / Assert
        error_msg = re.escape(
            "Invalid sdtypes: ['unexpected']. If you are trying to use a "
            'premium sdtype, contact info@sdv.dev about RDT Add-Ons.'
        )
        with pytest.raises(Error, match=error_msg):
            HyperTransformer._validate_config(config)

    def test__validate_config_invalid_transformer(self):
        """Test the ``_validate_config`` method.

        The method should crash if ``transformers`` is passed non-supported values.

        Input:
            - A config with incorrect ``transformers``.

        Expected behavior:
            - It should raise an error.
        """
        # Setup
        sdtypes = {
            'column1': 'numerical',
            'column2': 'numerical'
        }
        transformers = {
            'column1': FloatFormatter(),
            'column2': 'unexpected'
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers,
        }

        # Run / Assert
        error_msg = re.escape(
            "Invalid transformers for columns: ['column2']. "
            'Please assign an rdt transformer object to each column name.'
        )
        with pytest.raises(Error, match=error_msg):
            HyperTransformer._validate_config(config)

    def test_get_config(self):
        """Test the ``get_config`` method.

        The method should return a dictionary containing the following keys:
            - sdtypes: Maps to a dictionary that maps column names to ``sdtypes``.
            - transformers: Maps to a dictionary that maps column names to transformers.

        Setup:
            - Add entries to the ``field_sdtypes`` attribute.
            - Add entries to the ``field_transformers`` attribute.

        Output:
            - A dictionary with the key sdtypes mapping to the ``field_sdtypes`` attribute and
            the key transformers mapping to the ``field_transformers`` attribute.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        ht.field_sdtypes = {
            'column1': 'numerical',
            'column2': 'categorical'
        }

        # Run
        config = ht.get_config()

        # Assert
        expected_config = {
            'sdtypes': ht.field_sdtypes,
            'transformers': ht.field_transformers
        }
        assert config == expected_config

    def test_get_config_empty(self):
        """Test the ``get_config`` method when the config is empty.

        The method should return a dictionary containing the following keys:
            - sdtypes: Maps to a dictionary that maps column names to ``sdtypes``.
            - transformers: Maps to a dictionary that maps column names to transformers.

        Output:
            - A dictionary with the key sdtypes mapping to an empty dict and the key
            transformers mapping to an empty dict.
        """
        # Setup
        ht = HyperTransformer()

        # Run
        config = ht.get_config()

        # Assert
        expected_config = {
            'sdtypes': {},
            'transformers': {}
        }
        assert config == expected_config

    def test_set_config(self):
        """Test the ``set_config`` method.

        The method should set the ``instance.field_sdtypes``, and
        ``instance.field_transformers`` attributes based on the config.

        Setup:
            - Mock the ``_validate_config`` method so no warnings get raised.

        Input:
            - A dict with two keys:
                - transformers: Maps to a dict that maps column names to transformers.
                - sdtypes: Maps to a dict that maps column names to ``sdtypes``.

        Expected behavior:
            - The attributes , ``instance.field_sdtypes`` and ``instance.field_transformers``
            should be set.
        """
        # Setup
        transformers = {
            'column1': FloatFormatter(),
            'column2': FrequencyEncoder()
        }
        sdtypes = {
            'column1': 'numerical',
            'column2': 'categorical'
        }
        config = {
            'sdtypes': sdtypes,
            'transformers': transformers
        }
        ht = HyperTransformer()
        ht._validate_config = Mock()

        # Run
        ht.set_config(config)

        # Assert
        ht._validate_config.assert_called_once_with(config)
        assert ht.field_transformers == config['transformers']
        assert ht.field_sdtypes == config['sdtypes']

    @patch('rdt.hyper_transformer.warnings')
    def test_set_config_already_fitted(self, mock_warnings):
        """Test the ``set_config`` method.

        The method should raise a warning if the ``HyperTransformer`` has already been fit.

        Setup:
            - Mock the ``_validate_config`` method.
            - Mock warnings to make sure user warning is raised.
            - Set ``instance._fitted`` to True.

        Input:
            - A dict of two empty dicts.

        Expected behavior:
            - Warning should be raised to user.
        """
        # Setup

        config = {
            'sdtypes': {},
            'transformers': {}
        }
        ht = HyperTransformer()
        ht._fitted = True
        ht._validate_config = Mock()

        # Run
        ht.set_config(config)

        # Assert
        expected_warnings_msg = (
            'For this change to take effect, please refit your data using '
            "'fit' or 'fit_transform'."
        )
        mock_warnings.warn.assert_called_once_with(expected_warnings_msg)

    def get_data(self):
        return pd.DataFrame({
            'integer': [1, 2, 1, 3],
            'float': [0.1, 0.2, 0.1, 0.1],
            'categorical': ['a', 'a', 'b', 'a'],
            'bool': [False, False, True, False],
            'datetime': pd.to_datetime(['2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'])
        })

    def get_transformed_data(self, drop=False):
        data = pd.DataFrame({
            'integer': [1, 2, 1, 3],
            'float': [0.1, 0.2, 0.1, 0.1],
            'categorical': ['a', 'a', 'b', 'a'],
            'bool': [False, False, True, False],
            'datetime': pd.to_datetime(['2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01']),
            'integer.out': ['1', '2', '1', '3'],
            'integer.out.value': [1, 2, 1, 3],
            'float.value': [0.1, 0.2, 0.1, 0.1],
            'categorical.value': [0.375, 0.375, 0.875, 0.375],
            'bool.value': [0.0, 0.0, 1.0, 0.0],
            'datetime.value': [
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

    def test__validate_detect_config_called(self):
        """Test the ``_validate_detect_config_called`` method.

        Tests that the ``_validate_detect_config_called`` method raises an error
        when no values are passed to ``field_transformers`` and ``field_sdtypes``.

        Expected behavior:
            - An error should be raised.
        """
        # Setup
        ht = HyperTransformer()
        error_msg = (
            "No config detected. Set the config using 'set_config' or pre-populate "
            "it automatically from your data using 'detect_initial_config' prior to "
            'fitting your data.'
        )

        # Run / Assert
        with pytest.raises(Error, match=error_msg):
            ht._validate_detect_config_called(pd.DataFrame())

    def test__validate_detect_config_called_incorrect_data(self):
        """Test the ``_validate_detect_config_called`` method.

        Setup:
            - Mock the ``_validate_config_exists`` method.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht.field_sdtypes = {'col1': 'float', 'col2': 'categorical'}
        data = pd.DataFrame({'col1': [1, 2], 'col3': ['a', 'b']})
        error_msg = re.escape(
            'The data you are trying to fit has different columns than the original '
            "detected data (unknown columns: ['col3']). Column names and their "
            "sdtypes must be the same. Use the method 'get_config()' to see the expected "
            'values.'
        )

        # Run / Assert
        with pytest.raises(Error, match=error_msg):
            ht._validate_detect_config_called(data)

    def test__validate_detect_config_called_missing_columns(self):
        """Test the ``_validate_detect_config_called`` method.

        Tests that the ``_validate_detect_config_called`` method raises a ``NotFittedError``
        if any column names passed to ``fit`` are not present in the ones passed to
        ``detect_initial_config``.

        Expected behavior:
            - A ``NotFittedError`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_sdtypes = {'col1': 'float', 'col2': 'categorical'}
        data = pd.DataFrame({'col1': [1, 2]})
        error_msg = re.escape(
            'The data you are trying to fit has different columns than the original '
            'detected data. Column names and their sdtypes must be the same. Use the '
            "method 'get_config()' to see the expected values."
        )

        # Run / Assert
        with pytest.raises(Error, match=error_msg):
            ht._validate_detect_config_called(data)

    def test_fit(self):
        """Test the ``fit`` method.

        Tests that the ``fit`` method loops through the fields in ``field_transformers``
        and ``field_sdtypes`` that are in the data.

        Setup:
            - A mock for ``_fit_field_transformer``.
            - A mock for ``_field_in_set``.
            - A mock for ``_validate_detect_config_called``.

        Input:
            - A DataFrame with multiple columns of different sdtypes.

        Expected behavior:
            - The ``_fit_field_transformer`` mock should be called with the correct
            arguments in the correct order.
        """
        # Setup
        int_transformer = Mock()
        int_out_transformer = Mock()
        float_transformer = Mock()
        categorical_transformer = Mock()
        bool_transformer = Mock()
        datetime_transformer = Mock()

        data = self.get_data()
        field_transformers = {
            'integer': int_transformer,
            'float': float_transformer,
            'integer.out': int_out_transformer,
            'bool': bool_transformer,
            'categorical': categorical_transformer,
            'datetime': datetime_transformer
        }

        ht = HyperTransformer()
        ht.field_transformers = field_transformers
        ht._fit_field_transformer = Mock()
        ht._fit_field_transformer.return_value = data
        ht._field_in_set = Mock()
        ht._field_in_set.side_effect = [True, True, False, False, False]
        ht._validate_all_fields_fitted = Mock()
        ht._sort_output_columns = Mock()
        ht._validate_detect_config_called = Mock()
        ht._unfit = Mock()

        # Run
        ht.fit(data)

        # Assert
        ht._fit_field_transformer.call_args_list == [
            call(data, 'integer', int_transformer),
            call(data, 'float', float_transformer),
            call(data, 'categorical', categorical_transformer),
            call(data, 'bool', bool_transformer),
            call(data, 'datetime', datetime_transformer)
        ]
        ht._validate_all_fields_fitted.assert_called_once()
        ht._sort_output_columns.assert_called_once()
        ht._validate_detect_config_called.assert_called_once()
        ht._unfit.assert_called_once()

    def test_transform(self):
        """Test the ``transform`` method.

        Tests that ``transform`` loops through the ``_transformers_sequence``
        and calls ``transformer.transform`` in the correct order.

        Setup:
            - The ``_transformers_sequence`` will be hardcoded with a list
            of transformer mocks.
            - The ``_input_columns`` will be hardcoded.
            - The ``_output_columns`` will be hardcoded.
            - The ``_fitted`` attribute will be set to True.
            - The ``_validate_detect_config_called`` method will be mocked.

        Input:
            - A DataFrame of multiple sdtypes.

        Output:
            - The transformed DataFrame with the correct columns dropped.
        """
        # Setup
        int_transformer = Mock()
        int_out_transformer = Mock()
        float_transformer = Mock()
        categorical_transformer = Mock()
        bool_transformer = Mock()
        datetime_transformer = Mock()
        data = self.get_data()
        transformed_data = self.get_transformed_data()
        datetime_transformer.transform.return_value = transformed_data
        ht = HyperTransformer()
        ht._validate_detect_config_called = Mock()
        ht._validate_detect_config_called.return_value = True
        ht._fitted = True
        ht._transformers_sequence = [
            int_transformer,
            int_out_transformer,
            float_transformer,
            categorical_transformer,
            bool_transformer,
            datetime_transformer
        ]
        ht.field_sdtypes = {'col1': 'categorical'}
        ht._input_columns = list(data.columns)
        expected = self.get_transformed_data(True)
        ht._output_columns = list(expected.columns)

        # Run
        transformed = ht.transform(data)

        # Assert
        pd.testing.assert_frame_equal(transformed, expected)
        int_transformer.transform.assert_called_once()
        int_out_transformer.transform.assert_called_once()
        float_transformer.transform.assert_called_once()
        categorical_transformer.transform.assert_called_once()
        bool_transformer.transform.assert_called_once()
        datetime_transformer.transform.assert_called_once()

    def test_transform_raises_error_no_config(self):
        """Test that ``transform`` raises an error.

        The ``transform`` method should raise a ``Error`` if the ``config`` is empty.

        Input:
            - A DataFrame of multiple sdtypes.

        Expected behavior:
            - A ``Error`` is raised.
        """
        # Setup
        data = self.get_data()
        ht = HyperTransformer()

        # Run
        expected_msg = ("No config detected. Set the config using 'set_config' or pre-populate "
                        "it automatically from your data using 'detect_initial_config' prior to "
                        'fitting your data.')
        with pytest.raises(Error, match=expected_msg):
            ht.transform(data)

    def test_transform_raises_error_if_not_fitted(self):
        """Test that ``transform`` raises an error.

        The ``transform`` method should raise a ``NotFittedError`` if the
        ``HyperTransformer`` was not fitted.

        Setup:
            - The ``_fitted`` attribute will be False.
            - The ``_validate_config_exists`` method will be mocked.

        Input:
            - A DataFrame of multiple sdtypes.

        Expected behavior:
            - A ``NotFittedError`` is raised.
        """
        # Setup
        data = self.get_data()
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = False

        # Run
        with pytest.raises(NotFittedError):
            ht.transform(data)

    def test_transform_with_subset(self):
        """Test the ``transform`` method with a subset of the data.

        Tests that the ``transform`` method raises a ``Error`` if any column names passed
        to ``fit`` are not in the ones passed to ``transform``.

        Setup:
            - Mock the ``_validate_config_exists`` method.
            - Set the ``_input_columns``.

        Input:
            - A dataframe with a subset of the fitted columns.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._input_columns = ['col1', 'col2']
        data = pd.DataFrame({'col1': [1, 2]})

        # Run / Assert
        expected_msg = re.escape(
            'The data you are trying to transform has different columns than the original '
            'data. Column names and their sdtypes must be the same. Use the method '
            "'get_config()' to see the expected values."
        )
        with pytest.raises(Error, match=expected_msg):
            ht.transform(data)

    def test_transform_with_unknown_columns(self):
        """Test the ``transform`` method unknown columns.

        Tests that the ``transform`` method raises a ``Error`` if any column names passed
        to ``transform`` weren't seen in the fit.

        Setup:
            - Mock the ``_validate_config_exists`` method.
            - Set the ``_input_columns``.

        Input:
            - A dataframe with unknown columns.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._input_columns = ['col1']
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Run / Assert
        expected_msg = re.escape(
            'The data you are trying to transform has different columns than the original '
            'data. Column names and their sdtypes must be the same. Use the method '
            "'get_config()' to see the expected values."
        )
        with pytest.raises(Error, match=expected_msg):
            ht.transform(data)

    def test_transform_subset(self):
        """Test the ``transform_subset`` method with a subset of the data.

        Setup:
            - Mock the ``_transform`` method.

        Input:
            - A dataframe with a subset of the fitted columns.

        Expected behavior:
            - ``_reverse_transform`` is called with the correct parameters.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_detect_config_called = Mock()
        ht._validate_detect_config_called.return_value = True
        ht._fitted = True
        ht._transform = Mock()
        data = pd.DataFrame({'col1': [1, 2]})

        # Run
        ht.transform_subset(data)

        # Assert
        ht._transform.assert_called_once_with(data, prevent_subset=False)

    def test_transform_subset_with_unknown_columns(self):
        """Test the ``transform_subset`` method unknown columns.

        Tests that the ``transform_subset`` method raises a ``Error`` if any column names passed
        to ``transform`` weren't seen in the fit.

        Setup:
            - Mock the ``_validate_config_exists`` method.
            - Set the ``_input_columns``.

        Input:
            - A dataframe with unknown columns.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._input_columns = ['col1']
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Run / Assert
        expected_msg = re.escape(
            "Unexpected column names in the data you are trying to transform: ['col2']. "
            "Use 'get_config()' to see the acceptable column names."
        )
        with pytest.raises(Error, match=expected_msg):
            ht.transform_subset(data)

    def test_fit_transform(self):
        """Test call fit_transform"""
        # Run
        transformer = Mock()

        HyperTransformer.fit_transform(transformer, pd.DataFrame())

        # Asserts
        expect_call_count_fit = 1
        expect_call_count_transform = 1
        expect_call_args_fit = pd.DataFrame()
        expect_call_args_transform = pd.DataFrame()

        assert transformer.fit.call_count == expect_call_count_fit
        pd.testing.assert_frame_equal(
            transformer.fit.call_args[0][0],
            expect_call_args_fit
        )

        assert transformer.transform.call_count == expect_call_count_transform
        pd.testing.assert_frame_equal(
            transformer.transform.call_args[0][0],
            expect_call_args_transform
        )

    def test_reverse_transform(self):
        """Test the ``reverse_transform`` method.

        Tests that ``reverse_transform`` loops through the ``_transformers_sequence``
        in reverse order and calls ``transformer.reverse_transform``.

        Setup:
            - The ``_transformers_sequence`` will be hardcoded with a list
            of transformer mocks.
            - The ``_output_columns`` will be hardcoded.
            - The ``_input_columns`` will be hardcoded.

        Input:
            - A DataFrame of multiple sdtypes.

        Output:
            - The reverse transformed DataFrame with the correct columns dropped.
        """
        # Setup
        int_transformer = Mock()
        int_out_transformer = Mock()
        float_transformer = Mock()
        categorical_transformer = Mock()
        bool_transformer = Mock()
        datetime_transformer = Mock()
        data = self.get_transformed_data(True)
        reverse_transformed_data = self.get_transformed_data()
        int_transformer.reverse_transform.return_value = reverse_transformed_data
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._transformers_sequence = [
            int_transformer,
            int_out_transformer,
            float_transformer,
            categorical_transformer,
            bool_transformer,
            datetime_transformer
        ]
        ht._output_columns = list(data.columns)
        expected = self.get_data()
        ht._input_columns = list(expected.columns)

        # Run
        reverse_transformed = ht.reverse_transform(data)

        # Assert
        pd.testing.assert_frame_equal(reverse_transformed, expected)
        int_transformer.reverse_transform.assert_called_once()
        int_out_transformer.reverse_transform.assert_called_once()
        float_transformer.reverse_transform.assert_called_once()
        categorical_transformer.reverse_transform.assert_called_once()
        bool_transformer.reverse_transform.assert_called_once()
        datetime_transformer.reverse_transform.assert_called_once()

    def test_reverse_transform_raises_error_no_config(self):
        """Test that ``reverse_transform`` raises an error.

        The ``reverse_transform`` method should raise a ``Error`` if the config is empty.

        Input:
            - A DataFrame of multiple sdtypes.

        Expected behavior:
            - A ``NotFittedError`` is raised.
        """
        # Setup
        data = self.get_transformed_data()
        ht = HyperTransformer()

        # Run
        expected_msg = ("No config detected. Set the config using 'set_config' or pre-populate "
                        "it automatically from your data using 'detect_initial_config' prior to "
                        'fitting your data.')
        with pytest.raises(Error, match=expected_msg):
            ht.reverse_transform(data)

    def test_reverse_transform_raises_error_if_not_fitted(self):
        """Test that ``reverse_transform`` raises an error.

        The ``reverse_transform`` method should raise a ``NotFittedError`` if the
        ``_transformers_sequence`` is empty.

        Setup:
            - The ``_fitted`` attribute will be False.
            - Mock the ``_validate_config_exists`` method.

        Input:
            - A DataFrame of multiple sdtypes.

        Expected behavior:
            - A ``NotFittedError`` is raised.
        """
        # Setup
        data = self.get_transformed_data()
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = False

        # Run
        with pytest.raises(NotFittedError):
            ht.reverse_transform(data)

    def test_reverse_transform_with_subset(self):
        """Test the ``reverse_transform`` method with a subset of the data.

        Tests that the ``reverse_transform`` method raises a ``Error`` if any column names
        generated in ``transform`` are not in the data.

        Setup:
            - Mock the ``_validate_config_exists`` method.
            - Set the ``_output_columns``.

        Input:
            - A dataframe with a subset of the transformed columns.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._output_columns = ['col1', 'col2']
        data = pd.DataFrame({'col1': [1, 2]})

        # Run / Assert
        expected_msg = (
            'There are unexpected columns in the data you are trying to transform. You must '
            'provide a transformed dataset with all the columns from the original data.'
        )
        with pytest.raises(Error, match=expected_msg):
            ht.reverse_transform(data)

    def test_reverse_transform_with_unknown_columns(self):
        """Test the ``reverse_transform`` method with unknown columns.

        Tests that the ``reverse_transform`` method raises a ``Error`` if any column names
        in the data aren't expected in ``_output_columns``.

        Setup:
            - Mock the ``_validate_config_exists`` method.
            - Set the ``_output_columns``.

        Input:
            - A dataframe with an unseen column.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._output_columns = ['col1']
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Run / Assert
        expected_msg = (
            'There are unexpected columns in the data you are trying to transform. You must '
            'provide a transformed dataset with all the columns from the original data.'
        )
        with pytest.raises(Error, match=expected_msg):
            ht.reverse_transform(data)

    def test_reverse_transform_subset(self):
        """Test the ``reverse_transform_subset`` method with a subset of the data.

        Setup:
            - Mock the ``_validate_detect_config_called`` method.
            - Mock the ``_reverse_transform`` method.

        Input:
            - A dataframe with a subset of the transformed columns.

        Expected behavior:
            - ``_reverse_transform`` called with the right parameters.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_detect_config_called = Mock()
        ht._validate_detect_config_called.return_value = True
        ht._fitted = True
        ht._reverse_transform = Mock()
        data = pd.DataFrame({'col1.value': [1, 2]})

        # Run
        ht.reverse_transform_subset(data)

        # Assert
        ht._reverse_transform.assert_called_once_with(data, prevent_subset=False)

    def test_reverse_transform_subset_with_unknown_columns(self):
        """Test the ``reverse_transform_subset`` method with unknown columns.

        Tests that the ``reverse_transform_subset`` method raises a ``Error`` if any column names
        in the data aren't expected in ``_output_columns``.

        Setup:
            - Mock the ``_validate_config_exists`` method.
            - Set the ``_output_columns``.

        Input:
            - A dataframe with an unseen column.

        Expected behavior:
            - A ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._validate_config_exists = Mock()
        ht._validate_config_exists.return_value = True
        ht._fitted = True
        ht._output_columns = ['col1']
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Run / Assert
        expected_msg = re.escape(
            'There are unexpected column names in the data you are trying to transform. '
            "A reverse transform is not defined for ['col2']."
        )
        with pytest.raises(Error, match=expected_msg):
            ht.reverse_transform_subset(data)

    def test_update_transformers_by_sdtype_no_config(self):
        """Test that ``update_transformers_by_sdtype`` raises error if config is empty.

        Ensure that no changes have been made and an error message has been printed.

        Setup:
            - HyperTransformer instance.

        Side Effects:
            - HyperTransformer's field_transformers have not been upated.
            - Print has been called once with the expected error message.
        """
        # Setup
        ht = HyperTransformer()

        # Run
        expected_msg = (
            'Nothing to update. Use the `detect_initial_config` method to '
            'pre-populate all the sdtypes and transformers from your dataset.'
        )
        with pytest.raises(Error, match=expected_msg):
            ht.update_transformers_by_sdtype('categorical', object())

        # Assert
        assert ht.field_transformers == {}

    def test_update_transformers_by_sdtype_field_sdtypes_not_fitted(self):
        """Test ``update_transformers_by_sdtype`` if ``HyperTransformer`` hasn't been fitted.

        Ensure that the ``field_transformers`` matching the input ``sdtype`` have been updated.

        Setup:
            - HyperTransformer instance with ``field_transformers`` and ``field-data_types``.

        Side Effects:
            - HyperTransformer's ``field_transformers`` are upated with the input data.
            - Print not called.
        """
        # Setup
        ht = HyperTransformer()
        ff = FloatFormatter()
        ht.field_transformers = {
            'categorical_column': FrequencyEncoder(),
            'numerical_column': ff,
        }
        ht.field_sdtypes = {
            'categorical_column': 'categorical',
            'numerical_column': 'numerical',

        }
        transformer = LabelEncoder()

        # Run
        ht.update_transformers_by_sdtype('categorical', transformer)

        # Assert
        expected_field_transformers = {
            'categorical_column': transformer,
            'numerical_column': ff,
        }
        assert ht.field_transformers == expected_field_transformers

    def test_update_transformers_by_sdtype_with_transformer_as_none(self):
        """Test ``update_transformers_by_sdtype`` with ``transformer`` as ``None``.

        Ensure that ``update_transformers_by_sdtype`` works with ``transformer`` as ``None``.

        Setup:
            - HyperTransformer instance with ``field_transformers`` and ``field-data_types``.

        Side Effects:
            - HyperTransformer's ``field_transformers`` are upated with the ``None`` value.
        """
        # Setup
        ht = HyperTransformer()
        ff = FloatFormatter()
        ht.field_transformers = {
            'categorical_column': FrequencyEncoder(),
            'numerical_column': ff,
        }
        ht.field_sdtypes = {
            'categorical_column': 'categorical',
            'numerical_column': 'numerical',

        }

        # Run
        ht.update_transformers_by_sdtype('categorical', None)

        # Assert
        expected_field_transformers = {
            'categorical_column': None,
            'numerical_column': ff,
        }
        assert ht.field_transformers == expected_field_transformers

    @patch('rdt.hyper_transformer.warnings')
    def test_update_transformers_by_sdtype_field_sdtypes_fitted(self, mock_warnings):
        """Test ``update_transformers_by_sdtype`` if ``HyperTransformer`` has aleady been fit.

        Ensure that the ``field_transformers`` that have the input ``sdtype`` have been updated and
        a warning message has been raised.

        Setup:
            - HyperTransformer instance with ``field_transformers`` and ``field-data_types``.
            - ``instance._fitted`` set to True.

        Mock:
            - Warnings from the HyperTransformer.

        Side Effects:
            - HyperTransformer's ``field_transformers`` are upated with the input data.
        """
        # Setup
        ht = HyperTransformer()
        ht._fitted = True
        ht.field_transformers = {'categorical_column': FrequencyEncoder()}
        ht.field_sdtypes = {'categorical_column': 'categorical'}
        transformer = LabelEncoder()

        # Run
        ht.update_transformers_by_sdtype('categorical', transformer)

        # Assert
        expected_warnings_msg = (
            'For this change to take effect, please refit your data using '
            "'fit' or 'fit_transform'."
        )

        mock_warnings.warn.assert_called_once_with(expected_warnings_msg)
        assert ht.field_transformers == {'categorical_column': transformer}

    def test_update_transformers_by_sdtype_unsupported_sdtype_raises_error(self):
        """Test ``update_transformers_by_sdtype`` with an unsupported sdtype.

        An error should be raised.

        Setup:
            - HyperTransformer instance with ``field_transformers`` and ``field-data_types``.

        Side Effects:
            - Error is raised with a message about invalid sdtypes and premium sdtypes.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'categorical_column': Mock(),
            'numerical_column': Mock(),
        }
        ht.field_sdtypes = {
            'categorical_column': 'categorical',
            'numerical_column': 'numerical',
        }

        # Run / Assert
        expected_msg = (
            'Invalid sdtype. If you are trying to use a premium sdtype, contact info@sdv.dev '
            'about RDT Add-Ons.'
        )
        with pytest.raises(Error, match=expected_msg):
            ht.update_transformers_by_sdtype('fake_type', Mock())

    def test_update_transformers_by_sdtype_bad_transformer_raises_error(self):
        """Test ``update_transformers_by_sdtype`` with an object that isn't a transformer instance.

        Setup:
            - HyperTransformer instance with ``field_transformers`` and ``field-data_types``.

        Side Effects:
            - Error is raised with a message about using a transformer instance.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'categorical_column': Mock(),
            'numerical_column': Mock(),
        }
        ht.field_sdtypes = {
            'categorical_column': 'categorical',
            'numerical_column': 'numerical',
        }

        # Run / Assert
        expected_msg = 'Invalid transformer. Please input an rdt transformer object.'
        with pytest.raises(Error, match=expected_msg):
            ht.update_transformers_by_sdtype('categorical', Mock())

    def test_update_transformers_by_sdtype_mismatched_sdtype_raises_error(self):
        """Test ``update_transformers_by_sdtype`` with a mismatched sdtype and transformer.

        Setup:
            - HyperTransformer instance with ``field_transformers`` and ``field-data_types``.

        Side Effects:
            - Error is raised with a message about sdtype not matching transformer's sdtype.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'categorical_column': Mock(),
            'numerical_column': Mock(),
        }
        ht.field_sdtypes = {
            'categorical_column': 'categorical',
            'numerical_column': 'numerical',
        }

        # Run / Assert
        expected_msg = "The transformer you've assigned is incompatible with the sdtype."
        with pytest.raises(Error, match=expected_msg):
            ht.update_transformers_by_sdtype('categorical', FloatFormatter())

    @patch('rdt.hyper_transformer.warnings')
    def test_update_transformers_fitted(self, mock_warnings):
        """Test update transformers.

        Ensure that the function updates properly the ``self.field_transformers`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``True``.
            - Set some ``field_transformers``.

        Input:
            - Dictionary with a ``column_name`` and ``object()``.

        Mock:
            - Patch the ``warnings`` in order to ensure that expected message is being
              warn to the end user.
            - Transformer, mock for the transformer.

        Side Effects:
            - ``self.field_transformers`` has been updated.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = True
        instance.field_sdtypes = {'my_column': 'categorical'}
        instance.field_transformers = {'my_column': object()}
        instance._validate_transformers = Mock()
        transformer = FrequencyEncoder()
        column_name_to_transformer = {
            'my_column': transformer
        }

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        expected_message = (
            "For this change to take effect, please refit your data using 'fit' "
            "or 'fit_transform'."
        )

        mock_warnings.warn.assert_called_once_with(expected_message)
        assert instance.field_transformers['my_column'] == transformer
        instance._validate_transformers.assert_called_once_with(column_name_to_transformer)

    @patch('rdt.hyper_transformer.warnings')
    def test_update_transformers_not_fitted(self, mock_warnings):
        """Test update transformers.

        Ensure that the function updates properly the ``self.field_transformers`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``False``.
            - Set some ``field_transformers``.

        Input:
            - Dictionary with a ``column_name`` and a ``Mock`` transformer.

        Mock:
            - Patch the ``print`` function in order to ensure that expected message is being
              printed to the end user.
            - Transformer, mock for the transformer.

        Side Effects:
            - ``self.field_transformers`` has been updated.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        instance.field_transformers = {'my_column': BinaryEncoder}
        instance.field_sdtypes = {'my_column': 'boolean'}
        instance._validate_transformers = Mock()
        transformer = BinaryEncoder()
        column_name_to_transformer = {
            'my_column': transformer
        }

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        mock_warnings.warn.assert_not_called()
        assert instance.field_transformers['my_column'] == transformer
        instance._validate_transformers.assert_called_once_with(column_name_to_transformer)

    def test_update_transformers_no_field_transformers(self):
        """Test update transformers.

        Ensure that the function updates properly the ``self.field_transformers`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``False``.

        Input:
            - Dictionary with a ``column_name`` and a ``Mock`` transformer.

        Mock:
            - Patch the ``print`` function in order to ensure that expected message is being
              printed to the end user.
            - Transformer, mock for the replacement transformer.

        Side Effects:
            - ``self.field_transformers`` has been updated.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        mock_transformer = Mock()
        mock_transformer.get_input_sdtype.return_value = 'datetime'
        column_name_to_transformer = {
            'my_column': mock_transformer
        }

        # Run
        expected_msg = (
            'Nothing to update. Use the `detect_initial_config` method to pre-populate '
            'all the sdtypes and transformers from your dataset.'
        )
        with pytest.raises(Error, match=expected_msg):
            instance.update_transformers(column_name_to_transformer)

    @patch('rdt.hyper_transformer.print')
    def test_update_transformers_missmatch_sdtypes(self, mock_warnings):
        """Test update transformers.

        Ensure that the function updates properly the ``self.field_transformers`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer``.
            - Set ``field_transformers`` to contain a column and a mock transformer to be
              updated by ``update_transformer``.

        Input:
            - Dictionary with a ``column_name`` and a ``Mock`` transformer.

        Mock:
            - Patch the ``print`` function in order to ensure that expected message is being
              printed to the end user.
            - Transformer, mock for the replacement transformer.

        Side Effects:
            - ``self.field_transformers`` has been updated.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        mock_numerical = Mock()
        instance.field_transformers = {'my_column': mock_numerical}
        instance.field_sdtypes = {'my_column': 'categorical'}
        instance._validate_transformers = Mock()
        transformer = BinaryEncoder()
        column_name_to_transformer = {
            'my_column': transformer
        }

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        expected_call = (
            "Some transformers you've assigned are not compatible with the sdtypes. "
            f"Use 'update_sdtypes' to update: {'my_column'}"
        )

        assert mock_warnings.called_once_with(expected_call)
        instance._validate_transformers.assert_called_once_with(column_name_to_transformer)

    def test_update_transformers_transformer_is_none(self):
        """Test update transformers.

        Ensure that the function updates properly the ``self.field_transformers`` with ``None``.

        Setup:
            - Initialize ``HyperTransformer``.
            - Set ``field_transformers`` to contain a column and a mock transformer to be
              updated by ``update_transformer``.

        Input:
            - Dictionary with a ``column_name`` and a ``None`` transformer.

        Mock:
            - Transformer, mock for the replacement transformer.

        Side Effects:
            - ``self.field_transformers`` has been updated.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        mock_numerical = Mock()
        instance.field_transformers = {'my_column': mock_numerical}
        instance.field_sdtypes = {'my_column': 'categorical'}
        instance._validate_transformers = Mock()
        column_name_to_transformer = {
            'my_column': None
        }

        # Run
        instance.update_transformers(column_name_to_transformer)

        # Assert
        assert instance.field_transformers == {'my_column': None}
        instance._validate_transformers.assert_called_once_with(column_name_to_transformer)

    def test_update_transformers_column_doesnt_exist_in_config(self):
        """Test update transformers.

        Ensure that the function raises an error if the ``column`` is not in the
        ``self.field_transformers``.

        Setup:
            - Initialize ``HyperTransformer``.
            - Set ``field_transformers`` to contain a column and a mock transformer to be
              updated by ``update_transformer``.

        Input:
            - Dictionary with a ``column_name`` and a ``None`` transformer.

        Mock:
            - Transformer, mock for the replacement transformer.

        Side Effects:
            - An ``Error`` has been raised.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        mock_numerical = Mock()
        instance.field_transformers = {'my_column': mock_numerical}
        instance.field_sdtypes = {'my_column': 'categorical'}
        instance._validate_transformers = Mock()
        column_name_to_transformer = {
            'unknown_column': None
        }

        # Run / Assert
        expected_msg = re.escape(
            "Invalid column names: ['unknown_column']. These columns do not exist in "
            "the config. Use 'set_config()' to write and set your entire config at once."
        )
        with pytest.raises(Error, match=expected_msg):
            instance.update_transformers(column_name_to_transformer)

    @patch('rdt.hyper_transformer.warnings')
    def test_update_sdtypes_fitted(self, mock_warnings):
        """Test ``update_sdtypes``.

        Ensure that the method properly updates ``self.field_sdtypes`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``True``.
            - Set some ``field_sdtypes``.

        Input:
            - Dictionary with a ``column_name`` and ``sdtype``.

        Mock:
            - Patch the ``warnings`` module.
            - Mock the instance ``_user_message`` to ensure that has been called.

        Side Effects:
            - ``self.field_sdtypes`` has been updated.
            - Warning should be raised with the proper message.
        """
        # Setup
        instance = HyperTransformer()
        instance.field_transformers = {'a': FrequencyEncoder, 'b': FloatFormatter}
        instance.field_sdtypes = {'my_column': 'categorical'}
        instance._fitted = True
        instance._user_message = Mock()
        column_name_to_sdtype = {
            'my_column': 'numerical'
        }

        # Run
        instance.update_sdtypes(column_name_to_sdtype)

        # Assert
        expected_message = (
            "For this change to take effect, please refit your data using 'fit' "
            "or 'fit_transform'."
        )
        user_message = (
            'The transformers for these columns may change based on the new sdtype.\n'
            "Use 'get_config()' to verify the transformers."
        )

        mock_warnings.warn.assert_called_once_with(expected_message)
        assert instance.field_sdtypes == {'my_column': 'numerical'}
        instance._user_message.assert_called_once_with(user_message, 'Info')

    @patch('rdt.hyper_transformer.warnings')
    def test_update_sdtypes_not_fitted(self, mock_warnings):
        """Test ``update_sdtypes``.

        Ensure that the method properly updates the ``self.field_sdtypes`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``False``.
            - Set some ``field_sdtypes``.

        Input:
            - Dictionary with a ``column_name`` and ``sdtype``.

        Mock:
            - Patch the ``warnings`` module.

        Side Effects:
            - ``self.field_sdtypes`` has been updated.
            - No warning should be raised.
            - User message should be printed.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        instance._user_message = Mock()
        instance.field_sdtypes = {'my_column': 'categorical'}
        column_name_to_sdtype = {
            'my_column': 'numerical'
        }

        # Run
        instance.update_sdtypes(column_name_to_sdtype)

        # Assert
        user_message = (
            'The transformers for these columns may change based on the new sdtype.\n'
            "Use 'get_config()' to verify the transformers."
        )
        mock_warnings.warn.assert_not_called()
        assert instance.field_sdtypes == {'my_column': 'numerical'}
        instance._user_message.assert_called_once_with(user_message, 'Info')

    def test_update_sdtypes_no_field_sdtypes(self):
        """Test ``update_sdtypes``.

        Ensure that the method raises an error if there is no config.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``False`` and the
            ``field_sdtypes`` as empty.

        Input:
            - Dictionary with a ``column_name`` and ``sdtype``.

        Side Effects:
            - Error should be raised.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        instance.field_sdtypes = {}
        column_name_to_sdtype = {
            'my_column': 'numerical'
        }

        # Run / Assert
        expected_message = (
            'Nothing to update. Use the `detect_initial_config` method to pre-populate all the '
            'sdtypes and transformers from your dataset.'
        )
        with pytest.raises(Error, match=expected_message):
            instance.update_sdtypes(column_name_to_sdtype)

    def test_update_sdtypes_invalid_sdtype(self):
        """Test ``update_sdtypes``.

        Ensure that the method updates properly the ``self.field_sdtypes`` and prints the
        expected messages to guide the end-user.

        Setup:
            - Initialize ``HyperTransformer``.

        Input:
            - Dictionary with a ``column_name`` and invalid ``sdtype``.

        Side Effects:
            - Exception should be raised.
        """
        # Setup
        instance = HyperTransformer()
        instance._get_supported_sdtypes = Mock()
        instance._get_supported_sdtypes.return_value = []
        instance._fitted = False
        instance.field_sdtypes = {
            'my_column': 'categorical'
        }
        column_name_to_sdtype = {
            'my_column': 'credit_card'
        }

        # Run / Assert
        expected_message = re.escape(
            "Invalid sdtypes: ['credit_card']. If you are trying to use a "
            'premium sdtype, contact info@sdv.dev about RDT Add-Ons.'
        )
        with pytest.raises(Error, match=expected_message):
            instance.update_sdtypes(column_name_to_sdtype)

    def test_update_sdtypes_invalid_columns(self):
        """Test ``update_sdtypes``.

        Ensure that the method updates raises the appropriate error when passed
        columns not present in the config.

        Setup:
            - Initialize ``HyperTransformer``.

        Input:
            - Dictionary with an invalid ``column_name`` and ``sdtype``.

        Side Effects:
            - Exception should be raised.
        """
        # Setup
        instance = HyperTransformer()
        instance.field_sdtypes = {
            'my_column': 'categorical'
        }
        column_name_to_sdtype = {
            'unexpected': 'categorical'
        }

        # Run / Assert
        expected_message = re.escape(
            "Invalid column names: ['unexpected']. These columns do not exist in the "
            "config. Use 'set_config()' to write and set your entire config at once."
        )
        with pytest.raises(Error, match=expected_message):
            instance.update_sdtypes(column_name_to_sdtype)

    @patch('rdt.hyper_transformer.get_default_transformer')
    @patch('rdt.hyper_transformer.warnings')
    def test_update_sdtypes_different_sdtype(self, mock_warnings, default_mock):
        """Test ``update_sdtypes``.

        Ensure that the method properly updates the ``self.field_sdtypes`` and changes the
        transformer used if the ``sdtype`` is changed.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``False``.
            - Set some ``field_sdtypes``.

        Input:
            - Dictionary with a ``column_name`` and different ``sdtype``.

        Mock:
            - Patch the ``warnings`` module.
            - Patch the ``get_default_transformer`` method.

        Side Effects:
            - ``self.field_sdtypes`` has been updated.
            - ``self.field_transformers`` has been updated.
            - No warning should be raised.
            - User message should be printed.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        instance._user_message = Mock()
        instance.field_sdtypes = {'a': 'categorical'}
        transformer_mock = Mock()
        default_mock.return_value = transformer_mock
        column_name_to_sdtype = {
            'a': 'numerical'
        }

        # Run
        instance.update_sdtypes(column_name_to_sdtype)

        # Assert
        user_message = (
            'The transformers for these columns may change based on the new sdtype.\n'
            "Use 'get_config()' to verify the transformers."
        )
        mock_warnings.warn.assert_not_called()
        assert instance.field_sdtypes == {'a': 'numerical'}
        assert instance.field_transformers == {'a': transformer_mock}
        instance._user_message.assert_called_once_with(user_message, 'Info')

    @patch('rdt.hyper_transformer.warnings')
    def test_update_sdtypes_different_sdtype_transformer_provided(self, mock_warnings):
        """Test ``update_sdtypes``.

        Ensure that the method properly updates the ``self.field_sdtypes`` but doesn't change
        the transformer if it is already of the correct type.

        Setup:
            - Initialize ``HyperTransformer`` with ``_fitted`` as ``False``.
            - Set some ``field_sdtypes``.
            - Set ``field_transformers`` entry for the column whose ``sdtype`` will be changed
            to match the new ``sdtype``.

        Input:
            - Dictionary with a ``column_name`` and different ``sdtype``.

        Mock:
            - Patch the ``warnings`` module.

        Side Effects:
            - ``self.field_sdtypes`` has been updated.
            - ``self.field_transformers`` should not be updated.
            - No warning should be raised.
            - User message should be printed.
        """
        # Setup
        instance = HyperTransformer()
        instance._fitted = False
        instance._user_message = Mock()
        instance.field_sdtypes = {'a': 'categorical'}
        transformer = FloatFormatter()
        instance.field_transformers = {'a': transformer}
        column_name_to_sdtype = {
            'a': 'numerical'
        }

        # Run
        instance.update_sdtypes(column_name_to_sdtype)

        # Assert
        user_message = (
            'The transformers for these columns may change based on the new sdtype.\n'
            "Use 'get_config()' to verify the transformers."
        )
        mock_warnings.warn.assert_not_called()
        assert instance.field_sdtypes == {'a': 'numerical'}
        assert instance.field_transformers == {'a': transformer}
        instance._user_message.assert_called_once_with(user_message, 'Info')

    def test__validate_update_columns(self):
        """Test ``_validate_update_columns``.

        Ensure that the method properly raises an error when an invalid column is passed.

        Setup:
            - Initialize ``HyperTransformer``.

        Input:
            - List of ``update_columns``.
            - List of ``config_columns``.

        Side Effect:
            An error is raised with the columns that are not within the ``config_columns``.
        """
        # Setup
        instance = HyperTransformer()
        instance.field_sdtypes = {'col1': 'categorical'}

        # Run / Assert
        error_msg = re.escape(
            "Invalid column names: ['col2']. These columns do not exist in the "
            "config. Use 'set_config()' to write and set your entire config at once."
        )
        with pytest.raises(Error, match=error_msg):
            instance._validate_update_columns(['col1', 'col2'])

    def test__validate_transformers(self):
        """Test ``_validate_transformers``.

        Ensure that the method properly raises an error when an invalid transformer is passed.

        Setup:
            - Initialize ``HyperTransformer``.

        Input:
            - Dictionary of column names to transformers, where at least one of them is invalid.
        """
        # Setup
        instance = HyperTransformer()
        column_name_to_transformer = {
            'col1': FrequencyEncoder(),
            'col2': 'Unexpected',
            'col3': None
        }

        # Run / Assert
        error_msg = re.escape(
            "Invalid transformers for columns: ['col2']. "
            'Please assign an rdt transformer object to each column name.'
        )
        with pytest.raises(Error, match=error_msg):
            instance._validate_transformers(column_name_to_transformer)

    def test_remove_transformers(self):
        """Test the ``remove_transformers`` method.

        Test that the method removes the ``transformers`` from ``instance.field_transformers``.

        Setup:
            - Instance of HyperTransformers.
            - Set some ``field_transformers``.

        Input:
            - List with column name.

        Side Effects:
            - After removing using ``remove_transformers`` the columns transformer should be set
              to ``None``.
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'column1': 'transformer',
            'column2': 'transformer',
            'column3': 'transformer'
        }

        # Run
        ht.remove_transformers(column_names=['column2'])

        # Assert
        assert ht.field_transformers == {
            'column1': 'transformer',
            'column2': None,
            'column3': 'transformer'
        }

    def test_remove_transformers_unknown_columns(self):
        """Test the ``remove_transformers`` method.

        Test that the method raises an ``Error`` when a column does not exist in
        ``field_transformers``.

        Setup:
            - Instance of HyperTransformer.
            - Set some ``field_transformers``.

        Input:
            - List with column name, one valid and one invalid.

        Mock:
            - Mock ``warnings`` from ``rdt.hyper_transformer``.

        Side Effects:
            - When calling ``remove_transformers`` with a column that does not exist in the
              ``field_transformers`` an error should be raised..
        """
        # Setup
        ht = HyperTransformer()
        ht.field_transformers = {
            'column1': 'transformer',
            'column2': 'transformer',
            'column3': 'transformer'
        }

        error_msg = re.escape(
            "Invalid column names: ['column4']. These columns do not exist in the "
            "config. Use 'get_config()' to see the expected values."
        )

        # Run / Assert
        with pytest.raises(Error, match=error_msg):
            ht.remove_transformers(column_names=['column3', 'column4'])

        # Assert
        assert ht.field_transformers == {
            'column1': 'transformer',
            'column2': 'transformer',
            'column3': 'transformer'
        }

    @patch('rdt.hyper_transformer.warnings')
    def test_remove_transformers_fitted(self, mock_warnings):
        """Test the ``remove_transformers`` method.

        Test that the method raises a warning when changes have been applied to the
        ``field_transformers``.

        Setup:
            - Instance of HyperTransformer.
            - Set some ``field_transformers``.
            - Set ``ht._fitted`` to ``True``.

        Input:
            - List with column name.

        Mock:
            - Mock ``warnings`` from ``rdt.hyper_transformer``.

        Side Effects:
            - Warnings has been called once with the expected message.
        """
        # Setup
        ht = HyperTransformer()
        ht._fitted = True
        ht.field_transformers = {
            'column1': 'transformer',
            'column2': 'transformer',
            'column3': 'transformer'
        }

        # Run
        ht.remove_transformers(column_names=['column3', 'column2'])

        # Assert
        expected_warnings_msg = (
            'For this change to take effect, please refit your data using '
            "'fit' or 'fit_transform'."
        )
        mock_warnings.warn.assert_called_once_with(expected_warnings_msg)
        assert ht.field_transformers == {
            'column1': 'transformer',
            'column2': None,
            'column3': None
        }

    @patch('rdt.hyper_transformer.warnings')
    def test_remove_transformers_by_sdtype(self, mock_warnings):
        """Test remove transformers by sdtype.

        Test that when calling ``remove_transformers_by_sdtype``, those transformers
        that belong to this ``sdtype`` are being removed.

        Setup:
            - Instance of HyperTransformer.
            - Set ``field_transformers``.
            - Set ``field_sdtypes``.

        Input:
            - String representation for an sdtype.

        Side Effects:
            - ``instance.field_transformers`` are set to ``None`` for the columns
              that belong to that ``sdtype``.
        """
        # Setup
        ht = HyperTransformer()
        ht._fitted = True
        ht.field_transformers = {
            'column1': 'transformer',
            'column2': 'transformer',
            'column3': 'transformer'
        }
        ht.field_sdtypes = {
            'column1': 'numerical',
            'column2': 'categorical',
            'column3': 'categorical'
        }

        # Run
        ht.remove_transformers_by_sdtype('categorical')

        # Assert
        assert ht.field_transformers == {
            'column1': 'transformer',
            'column2': None,
            'column3': None
        }
        expected_warnings_msg = (
            'For this change to take effect, please refit your data using '
            "'fit' or 'fit_transform'."
        )
        mock_warnings.warn.assert_called_once_with(expected_warnings_msg)

    def test_remove_transformers_by_sdtype_premium_sdtype(self):
        """Test remove transformers by sdtype.

        Test that when calling ``remove_transformers_by_sdtype`` with a premium ``sdtype``
        an ``Error`` is being raised.

        Setup:
            - Instance of HyperTransformer.

        Input:
            - String representation for an sdtype.

        Side Effects:
            - When calling with a premium ``sdtype`` an ``Error`` should be raised.
        """
        # Setup
        ht = HyperTransformer()

        # Run
        error_msg = re.escape(
            "Invalid sdtype 'phone_number'. If you are trying to use a premium sdtype, "
            'contact info@sdv.dev about RDT Add-Ons.'
        )
        with pytest.raises(Error, match=error_msg):
            ht.remove_transformers_by_sdtype('phone_number')

    def test__get_transformer(self):
        """Test the ``_get_transformer`` method.

        The method should return the transformer instance stored for the field in the
        ``_transformers_tree``.

        Setup:
            - Set the ``_transformers_tree`` to have values for a few fields.
            - Set ``_fitted`` to ``True``.

        Input:
            - Each field name.

        Output:
            - The transformer instance used for each field.
        """
        # Setup
        ht = HyperTransformer()
        transformer1 = Mock()
        transformer2 = Mock()
        transformer3 = Mock()
        ht._transformers_tree = {
            'field': {'transformer': transformer1, 'outputs': ['field.out1', 'field.out2']},
            'field.out1': {'transformer': transformer2, 'outputs': ['field.out1.value']},
            'field.out2': {'transformer': transformer3, 'outputs': ['field.out2.value']}
        }
        ht._fitted = True

        # Run
        out1 = ht._get_transformer('field')
        out2 = ht._get_transformer('field.out1')
        out3 = ht._get_transformer('field.out2')

        # Assert
        assert out1 == transformer1
        assert out2 == transformer2
        assert out3 == transformer3

    def test__get_transformer_raises_error_if_not_fitted(self):
        """Test that ``_get_transformer`` raises ``NotFittedError``.

        If the ``HyperTransformer`` hasn't been fitted, a ``NotFittedError`` should be raised.

        Setup:
            - Set ``_fitted`` to ``False``.

        Expected behavior:
            - ``NotFittedError`` is raised.
        """
        # Setup
        ht = HyperTransformer()
        transformer1 = Mock()
        ht._transformers_tree = {
            'field1': {'transformer': transformer1, 'outputs': ['field1.out1', 'field1.out2']}
        }
        ht._fitted = False

        # Run / Assert
        with pytest.raises(NotFittedError):
            ht._get_transformer('field1')

    def test__get_output_transformers(self):
        """Test the ``_get_output_transformers`` method.

        The method should return a dict mapping each output column created from
        transforming the specified field, to the transformers to be used on them.

        Setup:
            - Set the ``_transformers_tree`` to have values for a few fields.
            - Set ``_fitted`` to ``False``.

        Output:
            - Dict mapping the outputs of the specified field to the transformers used on them.
        """
        # Setup
        ht = HyperTransformer()
        transformer1 = Mock()
        transformer2 = Mock()
        transformer3 = Mock()
        transformer4 = Mock()
        ht._transformers_tree = {
            'field1': {'transformer': transformer1, 'outputs': ['field1.out1', 'field1.out2']},
            'field1.out1': {'transformer': transformer2, 'outputs': ['field1.out1.value']},
            'field1.out2': {'transformer': transformer3, 'outputs': ['field1.out2.value']},
            'field2': {'transformer': transformer4, 'outputs': ['field2.value']}
        }
        ht._fitted = True

        # Run
        output_transformers = ht._get_output_transformers('field1')

        # Assert
        assert output_transformers == {
            'field1.out1': transformer2,
            'field1.out2': transformer3
        }

    def test__get_output_transformers_raises_error_if_not_fitted(self):
        """Test that the ``_get_output_transformers`` raises ``NotFittedError``.

        If the ``HyperTransformer`` hasn't been fitted, a ``NotFittedError`` should be raised.

        Setup:
            - Set ``_fitted`` to ``False``.

        Expected behavior:
            - ``NotFittedError`` is raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._fitted = False

        # Run / Assert
        with pytest.raises(NotFittedError):
            ht._get_output_transformers('field')

    def test__get_final_output_columns(self):
        """Test the ``_get_final_output_columns`` method.

        The method should traverse the tree starting at the provided field and return a list of
        all final column names (leaf nodes) that are descedants of that field.

        Setup:
            - Set the ``_transformers_tree`` to have some fields with multiple generations
            of descendants.
            - Set ``_fitted`` to ``False``.

        Output:
            - List of all column nodes with the specified field as an ancestor.
        """
        # Setup
        ht = HyperTransformer()
        transformer = Mock()
        ht._transformers_tree = {
            'field1': {
                'transformer': transformer,
                'outputs': ['field1.out1', 'field1.out2']
            },
            'field1.out1': {
                'transformer': transformer,
                'outputs': ['field1.out1.value']
            },
            'field1.out2': {
                'transformer': transformer,
                'outputs': ['field1.out2.out1', 'field1.out2.out2']
            },
            'field1.out2.out1': {
                'transformer': transformer,
                'outputs': ['field1.out2.out1.value']
            },
            'field1.out2.out2': {
                'transformer': transformer,
                'outputs': ['field1.out2.out2.value']
            },
            'field2': {
                'transformer': transformer,
                'outputs': ['field2.value']
            }
        }
        ht._fitted = True

        # Run
        outputs = ht._get_final_output_columns('field1')

        # Assert
        assert outputs == ['field1.out2.out2.value', 'field1.out2.out1.value', 'field1.out1.value']

    def test__get_final_output_columns_transformer_is_none(self):
        """Test the ``_get_final_output_columns`` method.

        Setup:
            - Set the ``_transformers_tree`` to have a field with a ``transformer`` as ``None``.

        Output:
            - List of the ``outputs`` that has been specified in the ``_transformers_tree``.
        """
        # Setup
        ht = HyperTransformer()
        ht._transformers_tree = {
            'field1': {
                'transformer': None,
                'outputs': ['field1']
            }
        }
        ht._fitted = True

        # Run
        outputs = ht._get_final_output_columns('field1')

        # Assert
        assert outputs == ['field1']

    def test__get_final_output_columns_raises_error_if_not_fitted(self):
        """Test that the ``_get_final_output_columns`` raises ``NotFittedError``.

        If the ``HyperTransformer`` hasn't been fitted, a ``NotFittedError`` should be raised.

        Setup:
            - Set ``_fitted`` to ``False``.

        Expected behavior:
            - ``NotFittedError`` is raised.
        """
        # Setup
        ht = HyperTransformer()
        ht._fitted = False

        # Run / Assert
        with pytest.raises(NotFittedError):
            ht._get_final_output_columns('field')

    def test__get_transformer_tree_yaml(self):
        """Test the ``_get_transformer_tree_yaml`` method.

        The method should return a YAML representation of the tree.

        Setup:
            - Set the ``_transformers_tree`` to have values for a few fields.

        Output:
            - YAML object rrepresenting tree.
        """
        # Setup
        ht = HyperTransformer()
        ht._transformers_tree = defaultdict(dict, {
            'field1': {
                'transformer': FrequencyEncoder(),
                'outputs': ['field1.out1', 'field1.out2']
            },
            'field1.out1': {
                'transformer': FloatFormatter(),
                'outputs': ['field1.out1.value']
            },
            'field1.out2': {
                'transformer': FloatFormatter(),
                'outputs': ['field1.out2.value']
            },
            'field2': {'transformer': FrequencyEncoder(), 'outputs': ['field2.value']}
        })
        ht._fitted = True

        # Run
        tree_yaml = ht._get_transformer_tree_yaml()

        # Assert
        assert tree_yaml == (
            'field1:\n  outputs:\n  - field1.out1\n  - field1.out2\n  '
            'transformer: FrequencyEncoder\nfield1.out1:\n  outputs:\n  - '
            'field1.out1.value\n  transformer: FloatFormatter\nfield1.out2:\n  '
            'outputs:\n  - field1.out2.value\n  transformer: FloatFormatter\nfield2:\n  '
            'outputs:\n  - field2.value\n  transformer: FrequencyEncoder\n'
        )
