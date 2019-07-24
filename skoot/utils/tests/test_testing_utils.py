# -*- coding: utf-8 -*-

from skoot.utils.testing import assert_raises, assert_persistable

import os


def func_that_raises():
    raise ValueError("Boo!")


def test_assert_raises():
    assert_raises(ValueError, func_that_raises)


def test_failing_assert_raises():
    def func_that_fails_assertion():
        assert_raises(ValueError, func=(lambda: None))

    assert_raises(AssertionError, func_that_fails_assertion)


def test_alternative_exception():
    def func_that_raises_type_error():
        raise TypeError("This is a type error!")

    def func_that_asserts_incorrectly():
        assert_raises(ValueError, func_that_raises_type_error)

    assert_raises(TypeError, func_that_raises_type_error)
    assert_raises(TypeError, func_that_asserts_incorrectly)


def test_fails_on_existing_location():
    loc = "exists.pkl"
    try:
        with open(loc, 'w') as tmp:
            tmp.write("Just creating a file so it raises")
        assert_raises(OSError, assert_persistable, None,
                      location=loc, X=None, y=None)

    finally:
        os.unlink(loc)
