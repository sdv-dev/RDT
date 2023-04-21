import sre_parse
from sre_constants import MAXREPEAT

from rdt.transformers.utils import _any, _max_repeat, strings_from_regex


def test_strings_from_regex_literal():
    generator, size = strings_from_regex('abcd', max_repeat=16)

    assert size == 1
    assert list(generator) == ['abcd']


def test_strings_from_regex_digit():
    generator, size = strings_from_regex('[0-9]')

    assert size == 10
    assert list(generator) == ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def test_strings_from_regex_repeat_literal():
    generator, size = strings_from_regex('a{1,3}')

    assert size == 3
    assert list(generator) == ['a', 'aa', 'aaa']


def test_strings_from_regex_repeat_digit():
    generator, size = strings_from_regex(r'\d{1,3}')

    assert size == 1110

    strings = list(generator)
    assert strings[0] == '0'
    assert strings[-1] == '999'


def test__any():
    options = {'a': 1}
    max_repeat = 5
    _any(options, max_repeat)


def test___max_repeat():
    options = (0, MAXREPEAT, [(sre_parse.LITERAL, 10)])
    _max_repeat(options, 16)


def test_strings_from_regex_very_large_regex():
    """Ensure that ``size`` of a very large regex is still computable."""
    very_large_regex = '[0-9a-zA-Z]{9}-[0-9a-zA-Z]{4}-[0-9a-zA-Z]{9}-[0-9a-zA-Z]{9}-[0-9a-z]{12}'
    generator, size = strings_from_regex(very_large_regex, max_repeat=16)

    assert size == 173689027553046619421110743915454114823342474255318764491341273608665169920
    [next(generator) for _ in range(100_000)]
