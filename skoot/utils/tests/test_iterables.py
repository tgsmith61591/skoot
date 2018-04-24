# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.utils.iterables import is_iterable, flatten_all, ensure_iterable
from sklearn.externals.six import u


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
