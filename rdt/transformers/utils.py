"""Utils for transformers."""


def get_subclasses(base):
    """Get all subclasses of a base class."""
    subclasses = base.__subclasses__()

    if len(subclasses) == 0:
        return []

    for subclass in subclasses:
        subclasses.extend(get_subclasses(subclass))

    return subclasses
