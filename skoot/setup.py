# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import

import os

from skoot._build_utils import maybe_cythonize_extensions


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('skoot', parent_package, top_path)

    # build utilities
    config.add_subpackage('__check_build')
    config.add_subpackage('__check_build/tests')
    config.add_subpackage('_build_utils')

    # modules
    config.add_subpackage('balance')
    config.add_subpackage('datasets')
    config.add_subpackage('decomposition')
    config.add_subpackage('feature_extraction')
    config.add_subpackage('feature_selection')
    config.add_subpackage('preprocessing')
    config.add_subpackage('testing')
    config.add_subpackage('utils')

    # module tests -- must be added after others!
    config.add_subpackage('balance/tests')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('decomposition/tests')
    config.add_subpackage('feature_extraction/tests')
    config.add_subpackage('feature_selection/tests')
    config.add_subpackage('preprocessing/tests')
    config.add_subpackage('testing/tests')
    config.add_subpackage('utils/tests')

    # do cythonization
    maybe_cythonize_extensions(top_path, config)

    # From Numpy doc:
    #   Generate package __config__.py file containing system_info
    #   information used during building the package.
    #
    #   This file is installed to the
    #   package installation directory.
    config.make_config_py()

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
