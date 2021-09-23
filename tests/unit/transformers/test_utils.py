from rdt.transformers.utils import get_subclasses


class TestBaseClass:
    pass


class TestChildClass(TestBaseClass):
    pass


class TestSecondChildClass(TestChildClass):
    pass


def test_get_subclasses():
    """Test the `utils.get_subclasses` method.

    Expect that get_subclasses returns all the subclasses
    of a class, and the subclasses of its subclasses.
    """
    subclasses = get_subclasses(TestBaseClass)

    assert len(subclasses) == 2
    assert TestChildClass in subclasses
    assert TestSecondChildClass in subclasses
