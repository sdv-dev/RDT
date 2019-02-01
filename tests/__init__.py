def safe_compare_dataframes(first, second):
    """Compare two dataframes even if they have NaN values.

    Args:
        first (pandas.DataFrame): DataFrame to compare
        second (pandas.DataFrame): DataFrame to compare

    Returns:
        bool
    """

    if first.isnull().all().all():
        return first.equals(second)

    else:
        nulls = (first.isnull() == second.isnull()).all().all()
        values = (first[~first.isnull()] == second[~second.isnull()]).all().all()
        return nulls and values
