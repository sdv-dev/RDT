"""Sdtype validation for multi-column transformers."""
from rdt.errors import TransformerInputError


class BaseValidation:
    """Base validation class."""

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
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and attr_name.startswith('_validate_'):
                attr(columns_to_sdtypes)


class AddressValidation(BaseValidation):
    """Validation class for Address data."""

    SUPPORTED_SDTYPES = [
        'country_code', 'administrative_unit', 'city', 'postcode',
        'street_address', 'secondary_address', 'state', 'state_abbr'
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


class GPSValidation(BaseValidation):
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
