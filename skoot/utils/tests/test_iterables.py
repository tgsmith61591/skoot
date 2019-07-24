# -*- coding: utf-8 -*-

from six import u
from skoot.utils.iterables import (chunk, is_iterable, flatten_all,
                                   ensure_iterable)

from skoot.utils.testing import assert_raises


def test_valid_is_iterable():
    # these are all non-string iterables
    assert is_iterable([1, 2, 3])
    assert is_iterable({1, 2, 3})
    assert is_iterable((1, 2, 3))


def test_invalid_is_iterable():
    assert not is_iterable('abc')
    assert not is_iterable(123)
    assert not is_iterable(u('1,2,3'))


def test_flatten():
    a = [[[], 3, 4], ['1', 'a'], [[[1]]], 1, 2]
    b = list(flatten_all(a))
    assert b == [3, 4, '1', 'a', 1, 1, 2], b


def test_ensure_iterable():
    x = 'a'
    assert ensure_iterable(x) == ['a']

    y = [1, 2, 3]
    assert ensure_iterable(y) is y


def test_chunking():
    def listify(chunks):
        return [list(c) for c in chunks]

    chunks = listify(chunk(range(11), 3))
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]

    # test same on known generator
    assert chunks == listify(chunk((i for i in range(11)), 3))

    # test corner where the input is of len 1
    assert listify(chunk([1], 1)) == [[1]]

    # this is the function that will fail
    failing_func = (lambda: list(chunk([1], 2)))
    assert_raises(ValueError, failing_func)
