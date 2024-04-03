import re
from unittest.mock import Mock, patch

import pytest

from rdt.errors import TransformerInputError
from rdt.transformers._validations import AddressValidator, BaseValidator, GPSValidator


class TestBaseValidator:

    @patch('rdt.transformers._validations.BaseValidator.SUPPORTED_SDTYPES', ['numerical'])
    @patch('rdt.transformers._validations.BaseValidator.VALIDATION_TYPE', 'Base')
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
        BaseValidator._validate_supported_sdtypes(columns_to_sdtypes_valid)
        with pytest.raises(TransformerInputError, match=expected_message):
            BaseValidator._validate_supported_sdtypes(columns_to_sdtypes_invalid)

    @patch('rdt.transformers._validations.BaseValidator._validate_supported_sdtypes')
    def test_validate_sdtypes(self, mock_validate_supported_sdtypes):
        """Test ``validate_sdtypes`` method."""
        # Setup
        columns_to_sdtypes = {
            'col1': 'numerical',
            'col2': 'categorical',
        }

        # Run
        BaseValidator.validate_sdtypes(columns_to_sdtypes)

        # Assert
        mock_validate_supported_sdtypes.assert_called_once_with(columns_to_sdtypes)

    def test_validate_imports(self):
        """Test ``validate_imports`` method."""
        # Run and Assert
        with pytest.raises(NotImplementedError):
            BaseValidator.validate_imports()


class TestAddressValidator:
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
        AddressValidator._validate_number_columns(columns_to_sdtypes_valid)

        expected_message = (
            'Address transformers takes up to 7 columns to transform. Please provide address'
            ' data with valid fields.'
        )
        with pytest.raises(TransformerInputError, match=re.escape(expected_message)):
            AddressValidator._validate_number_columns(column_to_sdtypes_invalid)

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
        AddressValidator._validate_uniqueness_sdtype(columns_to_sdtypes_valid)

        expected_message = re.escape(
            "Columns 'col_1', 'col_2' have the same sdtype 'country_code'.\n"
            "Columns 'col_3', 'col_4' have the same sdtype 'city'.\n"
            'Your address data cannot have duplicate fields.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            AddressValidator._validate_uniqueness_sdtype(columns_to_sdtypes_invalid)

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
        AddressValidator._validate_supported_sdtypes(columns_to_sdtypes_valid)

        expected_message = re.escape(
            "Column 'col_2' has an unsupported sdtype 'numerical'.\n"
            "Column 'col_3' has an unsupported sdtype 'categorical'.\n"
            'Please provide a column that is compatible with Address data.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            AddressValidator._validate_supported_sdtypes(columns_to_sdtypes_invalid)

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
        AddressValidator._validate_administrative_unit(columns_to_sdtypes_valid)

        expected_message = (
            "The AddressValidator can have up to 1 column with sdtype 'state'"
            " or 'administrative_unit'. Please provide address data with valid fields."
        )
        with pytest.raises(TransformerInputError, match=re.escape(expected_message)):
            AddressValidator._validate_administrative_unit(columns_to_sdtypes_invalid)

    def test__validate_sdtypes(self):
        """Test ``validate_sdtypes`` method."""
        # Setup
        columns_to_sdtypes = {
            'country': 'country_code',
            'region': 'administrative_unit',
        }
        AddressValidator._validate_number_columns = Mock()
        AddressValidator._validate_uniqueness_sdtype = Mock()
        AddressValidator._validate_supported_sdtypes = Mock()
        AddressValidator._validate_administrative_unit = Mock()

        # Run
        AddressValidator.validate_sdtypes(columns_to_sdtypes)

        # Assert
        AddressValidator._validate_number_columns.assert_called_once_with(columns_to_sdtypes)
        AddressValidator._validate_uniqueness_sdtype.assert_called_once_with(columns_to_sdtypes)
        AddressValidator._validate_supported_sdtypes.assert_called_once_with(columns_to_sdtypes)
        AddressValidator._validate_administrative_unit.assert_called_once_with(
            columns_to_sdtypes
        )

    def test__validate_imports_without_address_module(self):
        """Test ``validate_imports`` when address module doesn't exist."""
        # Run and Assert
        expected_message = (
            'You must have SDV Enterprise with the address add-on to use the address features'
        )
        with pytest.raises(ImportError, match=expected_message):
            AddressValidator.validate_imports()

    @patch('rdt.transformers')
    def test__validate_imports_without_premium_features(self, mock_transformers):
        """Test ``validate_imports`` when the user doesn't have the transformers."""
        # Setup
        mock_address = Mock()
        del mock_address.RandomLocationGenerator
        del mock_address.RegionalAnonymizer
        mock_transformers.address = mock_address

        # Run and Assert
        expected_message = (
            'You must have SDV Enterprise with the address add-on to use the address features'
        )
        with pytest.raises(ImportError, match=expected_message):
            AddressValidator.validate_imports()


class TestGPSValidator:
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
        GPSValidator._validate_uniqueness_sdtype(columns_to_sdtypes_valid)

        expected_message = re.escape(
            'The GPS columns must have one latitude and on longitude columns sdtypes. '
            'Please provide GPS data with valid fields.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            GPSValidator._validate_uniqueness_sdtype(columns_to_sdtypes_invalid)

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
        GPSValidator._validate_supported_sdtypes(columns_to_sdtypes_valid)

        expected_message = re.escape(
            "Column 'col_2' has an unsupported sdtype 'postal_code'.\n"
            'Please provide a column that is compatible with GPS data.'
        )
        with pytest.raises(TransformerInputError, match=expected_message):
            GPSValidator._validate_supported_sdtypes(columns_to_sdtypes_invalid)

    def test__validate_sdtypes(self):
        """Test ``validate_sdtypes`` method."""
        # Setup
        columns_to_sdtypes = {
            'latitude_column': 'latitude',
            'longitude_column': 'longitude',
        }
        GPSValidator._validate_uniqueness_sdtype = Mock()
        GPSValidator._validate_supported_sdtypes = Mock()

        # Run
        GPSValidator.validate_sdtypes(columns_to_sdtypes)

        # Assert
        GPSValidator._validate_uniqueness_sdtype.assert_called_once_with(columns_to_sdtypes)
        GPSValidator._validate_supported_sdtypes.assert_called_once_with(columns_to_sdtypes)

    def test_validate_import_gps_transformers_without_gps_module(self):
        """Test ``validate_imports`` when gps module doesn't exist."""
        # Run and Assert
        expected_message = (
            'You must have SDV Enterprise with the gps add-on to use the GPS features'
        )
        with pytest.raises(ImportError, match=expected_message):
            GPSValidator.validate_imports()

    @patch('rdt.transformers')
    def test_validate_import_gps_transformers_without_premium_features(self, mock_transformers):
        """Test ``validate_imports`` when the user doesn't have the transformers."""
        # Setup
        mock_gps = Mock()
        del mock_gps.RandomLocationGenerator
        del mock_gps.MetroAreaAnonymizer
        del mock_gps.GPSNoiser
        mock_transformers.gps = mock_gps

        # Run and Assert
        expected_message = (
            'You must have SDV Enterprise with the gps add-on to use the GPS features'
        )
        with pytest.raises(ImportError, match=expected_message):
            GPSValidator.validate_imports()
