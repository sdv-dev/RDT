"""RDT test module."""


def safe_compare_dataframes(first, second):
    """Compare two dataframes even if they have NaN values.

    Args:
        first (pandas.DataFrame): DataFrame to compare
        second (pandas.DataFrame): DataFrame to compare

    Returns:
        bool
    """
    if first.isna().all().all():
        return first.equals(second)

    else:
        nulls = (first.isna() == second.isna()).all().all()
        values = (first[~first.isna()] == second[~second.isna()]).all().all()
        return nulls and values
