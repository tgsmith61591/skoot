# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from scipy._build_utils import get_g77_abi_wrappers
    from skoot._build_utils.system_info import get_info
    from os.path import join

    config = Configuration('decomposition', parent_package, top_path)

    # Get lapack info for Fortran
    lapack_opt = get_info('lapack_opt')
    atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', [])
                      if k == 'ATLAS_INFO'] + [None])[0]
    if atlas_version:
        print('ATLAS version: %s' % atlas_version)
    else:
        print("No ATLAS version found")

    # fblas:
    # TODO:

    # get the src file glob, make a fortran lib
    dqrlib_src = [join('src/dqrlib', '*.f')]
    config.add_library('dqrlib', sources=dqrlib_src)

    # and now all .pyf sources:
    sources = ['dqrsl.pyf']
    config.add_extension(
        '_dqrsl',
        sources=sources,
        libraries=['dqrlib'],
        depends=dqrlib_src)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration().todict())
