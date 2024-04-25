"""Validations for multi-column transformers."""

import importlib

from rdt.errors import TransformerInputError


class BaseValidator:
    """Base validation class.

    The validation classes ensure that the input data is compatible with the transformers
    and that they can be imported.
    """

    SUPPORTED_SDTYPES = []
    VALIDATION_TYPE = None

    @classmethod
    def _validate_supported_sdtypes(cls, columns_to_sdtypes):
        message = ''
        for column, sdtype in columns_to_sdtypes.items():
            if sdtype not in cls.SUPPORTED_SDTYPES:
                message += f"Column '{column}' has an unsupported sdtype '{sdtype}'.\n"

        if message:
            message += (
                f'Please provide a column that is compatible with {cls.VALIDATION_TYPE} data.'
            )
            raise TransformerInputError(message)

    @classmethod
    def validate_sdtypes(cls, columns_to_sdtypes):
        """Validate the columns to sdtypes mapping.

        This method aims to call all other sdtype validation method in the class.

        Args:
            columns_to_sdtypes (dict):
                Mapping of column names to sdtypes.
        """
        raise NotImplementedError

    @classmethod
    def validate_imports(cls):
        """Check that the transformers can be imported."""
        raise NotImplementedError

    @classmethod
    def validate(cls, columns_to_sdtypes):
        """Validate the input data.

        Args:
            columns_to_sdtypes (dict):
                Mapping of column names to sdtypes.
        """
        cls.validate_sdtypes(columns_to_sdtypes)
        cls.validate_imports()


class AddressValidator(BaseValidator):
    """Validation class for Address data."""

    SUPPORTED_SDTYPES = [
        'country_code',
        'administrative_unit',
        'city',
        'postcode',
        'street_address',
        'secondary_address',
        'state',
        'state_abbr',
    ]
    VALIDATION_TYPE = 'Address'

    @classmethod
    def _validate_number_columns(cls, columns_to_sdtypes):
        if len(columns_to_sdtypes) > 7:
            raise TransformerInputError(
                f'{cls.VALIDATION_TYPE} transformers takes up to 7 columns to transform. '
                'Please provide address data with valid fields.'
            )

    @staticmethod
    def _validate_uniqueness_sdtype(columns_to_sdtypes):
        sdtypes_to_columns = {}
        for column, sdtype in columns_to_sdtypes.items():
            if sdtype not in sdtypes_to_columns:
                sdtypes_to_columns[sdtype] = []

            sdtypes_to_columns[sdtype].append(column)

        duplicate_fields = {
            value: keys for value, keys in sdtypes_to_columns.items() if len(keys) > 1
        }

        if duplicate_fields:
            message = ''
            for sdtype, columns in duplicate_fields.items():
                to_print = "', '".join(columns)
                message += f"Columns '{to_print}' have the same sdtype '{sdtype}'.\n"

            message += 'Your address data cannot have duplicate fields.'
            raise TransformerInputError(message)

    @classmethod
    def _validate_administrative_unit(cls, columns_to_sdtypes):
        num_column_administrative_unit = sum(
            1 for itm in columns_to_sdtypes.values() if itm in ['administrative_unit', 'state']
        )
        if num_column_administrative_unit > 1:
            raise TransformerInputError(
                f"The {cls.__name__} can have up to 1 column with sdtype 'state'"
                f" or 'administrative_unit'. Please provide address data with valid fields."
            )

    @classmethod
    def validate_sdtypes(cls, columns_to_sdtypes):
        """Validate the columns to sdtypes mapping."""
        cls._validate_supported_sdtypes(columns_to_sdtypes)
        cls._validate_number_columns(columns_to_sdtypes)
        cls._validate_uniqueness_sdtype(columns_to_sdtypes)
        cls._validate_administrative_unit(columns_to_sdtypes)

    @classmethod
    def validate_imports(cls):
        """Check that the address transformers can be imported."""
        error_message = (
            'You must have SDV Enterprise with the address add-on to use the address features.'
        )

        try:
            address_module = importlib.import_module('rdt.transformers.address')
        except ModuleNotFoundError:
            raise ImportError(error_message) from None

        required_classes = ['RandomLocationGenerator', 'RegionalAnonymizer']
        for class_name in required_classes:
            if not hasattr(address_module, class_name):
                raise ImportError(error_message)


class GPSValidator(BaseValidator):
    """Validation class for GPS data."""

    SUPPORTED_SDTYPES = ['latitude', 'longitude']
    VALIDATION_TYPE = 'GPS'

    @staticmethod
    def _validate_uniqueness_sdtype(columns_to_sdtypes):
        sdtypes_to_columns = {sdtype: column for column, sdtype in columns_to_sdtypes.items()}
        if len(sdtypes_to_columns) != 2:
            raise TransformerInputError(
                'The GPS columns must have one latitude and on longitude columns sdtypes. '
                'Please provide GPS data with valid fields.'
            )

    @classmethod
    def validate_sdtypes(cls, columns_to_sdtypes):
        """Validate the columns to sdtypes mapping."""
        cls._validate_supported_sdtypes(columns_to_sdtypes)
        cls._validate_uniqueness_sdtype(columns_to_sdtypes)

    @classmethod
    def validate_imports(cls):
        """Check that the GPS transformers can be imported."""
        error_message = 'You must have SDV Enterprise with the gps add-on to use the GPS features.'

        try:
            gps_module = importlib.import_module('rdt.transformers.gps')
        except ModuleNotFoundError:
            raise ImportError(error_message) from None

        required_classes = [
            'RandomLocationGenerator',
            'MetroAreaAnonymizer',
            'GPSNoiser',
        ]
        for class_name in required_classes:
            if not hasattr(gps_module, class_name):
                raise ImportError(error_message)
