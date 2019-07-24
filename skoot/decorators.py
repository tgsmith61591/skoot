# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from functools import wraps
import warnings

__all__ = [
    'overrides',
    'suppress_warnings'
]


def overrides(interface_class):
    """Decorator for methods that override super methods.

    This decorator provides runtime validation that the method is,
    in fact, inherited from the superclass. If not, will raise an
    ``AssertionError``.

    Parameters
    ----------
    interface_class : type
        The class/type from which the specified method is inherited.
        If the method does not exist in the specified type, a ``RuntimeError``
        will be raised.

    Examples
    --------
    The following is valid use:

        >>> class A(object):
        ...     def a(self):
        ...         return 1
        >>>
        >>> class B(A):
        ...     @overrides(A)
        ...     def a(self):
        ...         return 2
        ...
        ...     def b(self):
        ...         return 0

    The following would be an invalid ``overrides`` statement, since
    ``A`` does not have a ``b`` method to override.

        >>> class C(B): # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     @overrides(A) # should override B, not A
        ...     def b(self):
        ...         return 1
        Traceback (most recent call last):
        AssertionError: A.b must override a super method!
    """
    def overrider(method):
        assert (method.__name__ in dir(
            interface_class)), '%s.%s must override a super method!' % (
            interface_class.__name__, method.__name__)
        return method

    return overrider


def suppress_warnings(func):
    """Force a method to suppress all warnings it may raise.

    This decorator should be used with caution, as it may complicate
    debugging. For internal purposes, this is used for imports that cause
    consistent warnings (like pandas or matplotlib)

    Parameters
    ----------
    func : callable
        Automatically passed to the decorator. This
        function is run within the context of the warning
        filterer.

    Examples
    --------
    When any function is decorated with the ``suppress_warnings``
    decorator, any warnings that are raised will be suppressed.

        >>> import warnings
        >>>
        >>> @suppress_warnings
        ... def fun_that_warns():
        ...     warnings.warn("This is a warning", UserWarning)
        ...     return 1
        >>>
        >>> fun_that_warns()
        1
    """
    @wraps(func)
    def suppressor(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return suppressor
