# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, division, absolute_import


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError
    # from scipy._build_utils import get_g77_abi_wrappers, get_sgemv_fix
    from os.path import join

    config = Configuration('decomposition', parent_package, top_path)
    lapack_opt = get_info('lapack_opt')
    if not lapack_opt:
        raise NotFoundError("No lapack/BLAS resources found")

    # get the src file glob, make a fortran lib
    dqrlib_src = [join('src/dqrlib', '*.f')]
    config.add_library('dqrlib', sources=dqrlib_src)

    # and now all .pyf sources:
    sources = ['dqrsl.pyf']
    # sources += get_g77_abi_wrappers(lapack_opt)
    # sources += get_sgemv_fix(lapack_opt)
    print("Sources: %r" % sources)
    print("LAPACK: %r" % lapack_opt)

    config.add_extension(
        '_dqrsl',
        sources=sources,
        libraries=['dqrlib'],
        depends=dqrlib_src,
        extra_info=lapack_opt)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration().todict())
