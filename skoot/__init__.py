# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Sklearn-esque transformers and extension modules

import sys

__version__ = '1.0.0'

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of skoot when
    # the binaries are not built
    __SKOOT_SETUP__
except NameError:
    __SKOOT_SETUP__ = False

if __SKOOT_SETUP__:
    sys.stderr.write('Partial import of skoot during the build process.\n')
else:
    # check that the build completed properly. This prints an informative
    # message in the case that any of the C code was not properly compiled.
    from . import __check_build

    __all__ = [
        'balance',
        'datasets',
        'decomposition',
        'feature_selection',
        'preprocessing',
        'utils'
    ]


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
