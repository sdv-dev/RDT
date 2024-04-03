import re
from unittest.mock import Mock, patch

import pytest

from rdt.errors import TransformerInputError
from rdt.transformers._validations import AddressValidation, BaseValidation, GPSValidation


class TestBaseValidation:

    @patch('rdt.transformers._validations.BaseValidation.SUPPORTED_SDTYPES', ['numerical'])
    @patch('rdt.transformers._validations.BaseValidation.VALIDATION_TYPE', 'Base')
    def test_validate_supported_sdtypes(self):
        """Test ``_validate_supported_sdtypes`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col1': 'numerical',
            'col2': 'numerical',
        }
        columns_to_sdtypes_invalid = {
            'col1': 'numerical',
            'col2': 'categorical',
            'col3': 'categorical',
        }

        expected_message = re.escape(
            "Column 'col2' has an unsupported sdtype 'categorical'.\n"
            "Column 'col3' has an unsupported sdtype 'categorical'.\n"
            'Please provide a column that is compatible with Base data.'
        )

        # Run and Assert
        BaseValidation._validate_supported_sdtypes(columns_to_sdtypes_valid)
        with pytest.raises(TransformerInputError, match=expected_message):
            BaseValidation._validate_supported_sdtypes(columns_to_sdtypes_invalid)

    @patch('rdt.transformers._validations.BaseValidation._validate_supported_sdtypes')
    def test_validate_sdtypes(self, mock_validate_supported_sdtypes):
        """Test ``validate_sdtypes`` method."""
        # Setup
        columns_to_sdtypes = {
            'col1': 'numerical',
            'col2': 'categorical',
        }

        # Run
        BaseValidation.validate_sdtypes(columns_to_sdtypes)

        # Assert
        mock_validate_supported_sdtypes.assert_called_once_with(columns_to_sdtypes)


class TestAddressValidation:
    def test__validate_number_columns(self):
        """Test ``_validate_number_columns`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col_1': 'country_code',
            'col_2': 'administrative_unit',
        }
        column_to_sdtypes_invalid = {
            'col_1': 'country_code',
            'col_2': 'administrative_unit',
            'col_3': 'city',
            'col_4': 'postcode',
            'col_5': 'street_address',
            'col_6': 'secondary_address',
            'col_7': 'country_code',
            'col_8': 'administrative_unit'
        }

        # Run and Assert
        AddressValidation._validate_number_columns(columns_to_sdtypes_valid)

        expected_message = (
            'Address transformers takes up to 7 columns to transform. Please provide address'
            ' data with valid fields.'
        )
        with pytest.raises(TransformerInputError, match=re.escape(expected_message)):
            AddressValidation._validate_number_columns(column_to_sdtypes_invalid)

    def test__validate_uniqueness_sdtype(self):
        """Test ``_validate_uniqueness_sdtype`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col_1': 'country_code',
            'col_2': 'administrative_unit',
        }
        columns_to_sdtypes_invalid = {
            'col_1': 'country_code',
            'col_2': 'country_code',
            'col_3': 'city',
            'col_4': 'city'
        }

        # Run and Assert
        AddressValidation._validate_uniqueness_sdtype(columns_to_sdtypes_valid)

        expected_message = re.escape(
            "Columns 'col_1', 'col_2' have the same sdtype 'country_code'.\n"
            "Columns 'col_3', 'col_4' have the same sdtype 'city'.\n"
            'Your address data cannot have duplicate fields.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            AddressValidation._validate_uniqueness_sdtype(columns_to_sdtypes_invalid)

    def test__validate_supported_sdtype(self):
        """Test ``_validate_supported_sdtype`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col_1': 'country_code',
            'col_2': 'administrative_unit',
        }
        columns_to_sdtypes_invalid = {
            'col_1': 'country_code',
            'col_2': 'numerical',
            'col_3': 'categorical',
        }

        # Run and Assert
        AddressValidation._validate_supported_sdtypes(columns_to_sdtypes_valid)

        expected_message = re.escape(
            "Column 'col_2' has an unsupported sdtype 'numerical'.\n"
            "Column 'col_3' has an unsupported sdtype 'categorical'.\n"
            'Please provide a column that is compatible with Address data.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            AddressValidation._validate_supported_sdtypes(columns_to_sdtypes_invalid)

    def test__validate_administrative_unit(self):
        """Test ``_validate_administrative_unit`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col_1': 'country_code',
            'col_2': 'administrative_unit',
        }
        columns_to_sdtypes_invalid = {
            'col_1': 'administrative_unit',
            'col_2': 'state'
        }

        # Run and Assert
        AddressValidation._validate_administrative_unit(columns_to_sdtypes_valid)

        expected_message = (
            "The AddressValidation can have up to 1 column with sdtype 'state'"
            " or 'administrative_unit'. Please provide address data with valid fields."
        )
        with pytest.raises(TransformerInputError, match=re.escape(expected_message)):
            AddressValidation._validate_administrative_unit(columns_to_sdtypes_invalid)

    def test__validate_sdtypes(self):
        """Test ``validate_sdtypes`` method."""
        # Setup
        columns_to_sdtypes = {
            'country': 'country_code',
            'region': 'administrative_unit',
        }
        AddressValidation._validate_number_columns = Mock()
        AddressValidation._validate_uniqueness_sdtype = Mock()
        AddressValidation._validate_supported_sdtypes = Mock()
        AddressValidation._validate_administrative_unit = Mock()

        # Run
        AddressValidation.validate_sdtypes(columns_to_sdtypes)

        # Assert
        AddressValidation._validate_number_columns.assert_called_once_with(columns_to_sdtypes)
        AddressValidation._validate_uniqueness_sdtype.assert_called_once_with(columns_to_sdtypes)
        AddressValidation._validate_supported_sdtypes.assert_called_once_with(columns_to_sdtypes)
        AddressValidation._validate_administrative_unit.assert_called_once_with(
            columns_to_sdtypes
        )


class TestGPSValidation:
    def test__validate_uniqueness_sdtype(self):
        """Test ``_validate_uniqueness_sdtype`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col_1': 'latitude',
            'col_2': 'longitude',
        }
        columns_to_sdtypes_invalid = {
            'col_1': 'latitude',
            'col_2': 'latitude',
        }

        # Run and Assert
        GPSValidation._validate_uniqueness_sdtype(columns_to_sdtypes_valid)

        expected_message = re.escape(
            'The GPS columns must have one latitude and on longitude columns sdtypes. '
            'Please provide GPS data with valid fields.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            GPSValidation._validate_uniqueness_sdtype(columns_to_sdtypes_invalid)

    def test__validate_supported_sdtype(self):
        """Test ``_validate_supported_sdtype`` method."""
        # Setup
        columns_to_sdtypes_valid = {
            'col_1': 'latitude',
            'col_2': 'longitude',
        }
        columns_to_sdtypes_invalid = {
            'col_1': 'latitude',
            'col_2': 'postal_code',
        }

        # Run and Assert
        GPSValidation._validate_supported_sdtypes(columns_to_sdtypes_valid)

        expected_message = re.escape(
            "Column 'col_2' has an unsupported sdtype 'postal_code'.\n"
            'Please provide a column that is compatible with GPS data.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            GPSValidation._validate_supported_sdtypes(columns_to_sdtypes_invalid)

    def test__validate_sdtypes(self):
        """Test ``validate_sdtypes`` method."""
        # Setup
        columns_to_sdtypes = {
            'latitude_column': 'latitude',
            'longitude_column': 'longitude',
        }
        GPSValidation._validate_uniqueness_sdtype = Mock()
        GPSValidation._validate_supported_sdtypes = Mock()

        # Run
        GPSValidation.validate_sdtypes(columns_to_sdtypes)

        # Assert
        GPSValidation._validate_uniqueness_sdtype.assert_called_once_with(columns_to_sdtypes)
        GPSValidation._validate_supported_sdtypes.assert_called_once_with(columns_to_sdtypes)
