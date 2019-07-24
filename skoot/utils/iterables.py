# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import six
import types

from .compat import xrange

__all__ = [
    'chunk',
    'ensure_iterable',
    'flatten_all',
    'is_iterable'
]


def ensure_iterable(element):
    """Make an element an iterable.

    If an element is already iterable, return it as is. If it's not, return
    it inside of a list. This helper function allows us to avoid clunky if/then
    checks all over the place::

        if not is_iterable(this):
            this = [this]

    Parameters
    ----------
    element : object
        An iterable or not
    """
    if not is_iterable(element):
        element = [element]
    return element


def flatten_all(container):
    """Recursively flattens an arbitrarily nested iterable.

    Parameters
    ----------
    container : array_like, shape=(n_items,)
        The iterable to flatten. If the ``container`` is
        not iterable, it will be returned in a list as
        ``[container]``

    Examples
    --------
    The example below produces a list of mixed results:

        >>> a = [[[], 3, 4],['1', 'a'],[[[1]]], 1, 2]
        >>> list(flatten_all(a))
        [3, 4, '1', 'a', 1, 1, 2]

    Returns
    -------
    res : generator
        A generator of all of the flattened values.
    """
    if not is_iterable(container):
        yield container
    else:
        for i in container:
            if is_iterable(i):
                for j in flatten_all(i):
                    yield j
            else:
                yield i


def is_iterable(x):
    """Determine whether an element is iterable.

    This function determines whether an element is iterable by checking
    for the ``__iter__`` attribute. Since Python 3.x adds the ``__iter__``
    attribute to strings, we also have to make sure the input is not a
    string or unicode type.

    Parameters
    ----------
    x : object
        The object or primitive to test whether
        or not is an iterable.
    """
    if isinstance(x, six.string_types):
        return False
    return hasattr(x, '__iter__')


def chunk(v, n):
    """Chunk a vector into k roughly equal parts.

    Parameters
    ----------
    v : array-like, shape=(n_samples,)
        The vector of values.

    n : int
        The number of chunks to produce.
    """
    # if v is a generator, we need it as a list...
    if isinstance(v, types.GeneratorType):
        v = list(v)
    len_v = len(v)

    # fail out if n > len_v
    if n > len_v:
        raise ValueError("N exceeds length of vector!")

    k, m = divmod(len_v, n)
    for i in xrange(n):
        yield v[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
