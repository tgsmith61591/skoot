# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy as np

    config = Configuration('decomposition', parent_package, top_path)
    config.add_extension('dqrsl', sources=['dqrsl.f'],
                         include_dirs=[np.get_include()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration().todict())
