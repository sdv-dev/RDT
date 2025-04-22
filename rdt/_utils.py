from collections import defaultdict

from rdt.errors import InvalidConfigError


def _validate_unique_transformer_instances(column_name_to_transformer):
    """Validate that the transformer instance for each field is unique.

    Args:
        column_name_to_transformer (dict):
            A dictionary mapping column names to their current transformer.

    Raises:
        - ``InvalidConfigError`` if transformers in ``column_name_to_transformer`` are repeated.
    """
    seen_transformers = defaultdict(set)
    for column_name, transformer in column_name_to_transformer.items():
        if transformer is not None:
            seen_transformers[transformer].add(column_name)

    duplicated_transformers = {
        transformer: columns
        for transformer, columns in seen_transformers.items()
        if len(columns) > 1
    }
    if duplicated_transformers:
        duplicated_column_messages = []
        for duplicated_columns in duplicated_transformers.values():
            columns = ', '.join(
                sorted([
                    str(columns) if not isinstance(columns, str) else f"'{columns}'"
                    for columns in duplicated_columns
                ])
            )
            duplicated_column_messages.append(f'columns ({columns})')

        if len(duplicated_column_messages) > 1:
            plurality = 'instances are'
        else:
            plurality = 'instance is'

        error_message = (
            f'The same transformer {plurality} being assigned to '
            f'{", ".join(duplicated_column_messages)}. Please create different transformer objects '
            'for each assignment.'
        )
        raise InvalidConfigError(error_message)
