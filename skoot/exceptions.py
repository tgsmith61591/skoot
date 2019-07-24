# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

__all__ = [
    'DeveloperError'
]


class DeveloperError(BaseException):
    """Used to indicate a developer screwed up.

    This error is raised for soft coding requirements that cannot be
    enforced until runtime. If you see it, some developer somewhere
    screwed up. :-)
    """


class ValidationWarning(UserWarning):
    """Used to indicate a warning in the validation submodule."""
