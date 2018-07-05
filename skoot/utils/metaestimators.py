# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Estimators and decorators for estimators

from __future__ import absolute_import

from functools import wraps
import time

__all__ = [
    'timed_instance_method'
]


def timed_instance_method(attribute_name):
    """Function timer decorator.

    This function is used to decorate instance-based methods in
    ``BasePDTransformer`` classes. In particular, ``fit`` and ``transform``.
    After computing the result of the decorated method, the run time is bound
    to the instance under the attribute ``param_name``.

    Parameters
    ----------
    attribute_name : str or unicode
        The name of the attribute under which to save the runtime of the
        decorated method. This is bound to the instance class.
    """
    def method_wrapper(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            # The method must contain the "im_self" parameter. This means that
            # it's bound as an instance method. If it's not present, we'll have
            # to raise an ValueError since this was only designed for instance
            # methods in transformer classes. Note we cannot check this outside
            # of this inner function, since otherwise it will be evaluated at
            # definition time and not runtime (so at class def, not
            # instance def)
            if not hasattr(method, "im_self"):
                raise ValueError("'timed_instance_method' can only be used "
                                 "on instance methods, not staticmethods or "
                                 "standalone functions. Method %s does not "
                                 "appear to be an instance method (%r)"
                                 % (method.__name__, dir(method)))

            start_time = time.time()
            result = method(*args, **kwargs)

            # Bind the run time to the 'im_self' parameter of the method, which
            # points to the instance of the class
            setattr(method.im_self, attribute_name, time.time() - start_time)
            return result
        return wrapper
    return method_wrapper
