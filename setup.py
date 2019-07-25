# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup the skoot module
"""Skoot: Accelerate your data science pipelines

Skoot is an open-source python package for data scientists and
other machine learning practitioners who would like to spend less time
developing bespoke solutions for data pre-processing. The library aims to
tie together as many common techniques and transformers in one location
as possible, doing so with a familiar scikit-learn API feel.
"""

from __future__ import print_function, absolute_import, division

from distutils.command.clean import clean
import warnings
import shutil
import os
import sys

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# get the description out of the first line further down
DOCLINES = __doc__.split(os.linesep)

# Hacky (!!), adopted from sklearn & scipy. This sets a global variable
# so skoot __init__ can detect if it's being loaded in the setup
# routine, so it won't load submodules that haven't yet been built.
# This is because of the numpy distutils extensions that are used by
# skoot to build the compiled extensions in sub-packages
builtins.__SKOOT_SETUP__ = True

# metadata
DISTNAME = 'skoot'
PYPIDIST = DISTNAME
DESCRIPTION = DOCLINES[0]

MAINTAINER = 'Taylor G. Smith'
MAINTAINER_GIT = 'tgsmith61591'
MAINTAINER_EMAIL = 'taylor.smith@alkaline-ml.com'
LICENSE = 'MIT'

# import restricted version
import skoot
VERSION = skoot.__version__

# get the installation requirements:
with open('requirements.txt') as req:
    REQUIREMENTS = [l for l in req.read().split("\n") if l]
    print("Setup requirements: %s" % str(REQUIREMENTS))

SETUPTOOLS_COMMANDS = {  # this is a set literal, not a dict
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed'
}

# are we building from install or develop? Since "install" is not in the
# SETUPTOOLS_COMMANDS, we have to check that here...
we_be_buildin = 'install' in sys.argv

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    # we don't use setuptools, but if we don't import it, the "develop"
    # option for setup.py is invalid.
    import setuptools
    from setuptools.dist import Distribution

    class BinaryDistribution(Distribution):
        """Command class to indicate binary distribution.

        The goal is to avoid having to later build the C or Fortran code
        on the system itself, but to build the binary dist wheels on the
        CI platforms. This class helps us achieve just that.

        References
        ----------
        .. [1] How to avoid building a C library with my Python package:
               http://bit.ly/2vQkW47

        .. [2] https://github.com/spotify/dh-virtualenv/issues/113
        """
        def is_pure(self):
            """Return False (not pure).

            Since we are distributing binary (.so, .dll, .dylib) files for
            different platforms we need to make sure the wheel does not build
            without them! See 'Building Wheels':
            http://lucumr.pocoo.org/2014/1/27/python-on-wheels/
            """
            return False

        def has_ext_modules(self):
            """Return True (there are external modules).

            The package has external modules. Therefore, unsurprisingly,
            this returns True to indicate that there are, in fact, external
            modules.
            """
            return True

    # only import numpy (later) if we're developing
    if any(cmd in sys.argv for cmd in {'develop', 'bdist_wheel',
                                       'bdist_wininst'}):
        we_be_buildin = True

    print('Adding extra setuptools args')
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        package_data={DISTNAME: ['*']},
        distclass=BinaryDistribution
    )

    # if we have to build, we need to add cython to the requirements
    REQUIREMENTS.append("cython>=0.23")
else:
    extra_setuptools_args = dict()


# Custom clean command to remove build artifacts -- adopted from sklearn
class CleanCommand(clean):
    description = "Remove build artifacts from the source tree"

    # this is mostly in case we ever add a Cython module to SMRT
    def run(self):
        clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c & .so files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk(DISTNAME):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    print('Removing file: %s' % filename)
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                # latter is for FORTRAN modules...
                if dirname == '__pycache__' or dirname.endswith('.dSYM'):
                    print('Removing directory: %s' % dirname)
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}


def configuration(parent_package='', top_path=None):
    # we know numpy is a valid import now
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(DISTNAME)
    return config


def do_setup():
    # For non-build actions, NumPy is not required, so we can use the
    # setuptools module. However this is not preferable... the moment
    # setuptools is imported, it monkey-patches distutils' setup and
    # changes its behavior...
    # (https://github.com/scikit-learn/scikit-learn/issues/1016)
    # numpy might not be on the system yet
    from setuptools import setup

    # setup the config
    metadata = dict(name=PYPIDIST,
                    packages=[DISTNAME],
                    url="https://github.com/%s/%s" % (MAINTAINER_GIT, DISTNAME),
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    version=VERSION,
                    classifiers=[
                        'Programming Language :: C',
                        'Programming Language :: Fortran',
                        'Programming Language :: Python :: 3.5',
                        'Programming Language :: Python :: 3.6',
                        'Programming Language :: Python :: 3.7',
                        'Topic :: Software Development',
                        'Topic :: Scientific/Engineering',
                        'Operating System :: POSIX',
                        'Operating System :: Unix',
                    ],
                    keywords='scikit-learn pandas machine-learning '
                             'data-science',
                    # this will only work for releases that have the right tag
                    download_url='https://github.com/%s/%s/archive/v%s.tar.gz'
                                 % (MAINTAINER_GIT, DISTNAME, VERSION),
                    python_requires='>=3.5, <4',
                    platforms=["Windows", "Linux", "Unix", "Mac OS-X"],
                    cmdclass=cmdclass,
                    setup_requires=REQUIREMENTS,
                    install_requires=REQUIREMENTS,
                    **extra_setuptools_args)

    # if we are building for install, develop or bdist_wheel, we NEED
    # numpy and cython, since they are both used in building the .pyx
    # files into C modules.
    if we_be_buildin:
        try:
            # overwrites "setup" in the namespace
            from numpy.distutils.core import setup
        except ImportError:
            raise RuntimeError('Need numpy to build %s' % DISTNAME)

        # Cythonize and Fortranize should theoretically be delegated to
        # submodules' setup.py scripts... *theoretically*...

        # add the config to the metadata
        metadata['configuration'] = configuration

    # otherwise we're building from a wheel and non-build
    # actions don't require numpy
    else:
        metadata['version'] = VERSION

    # call setup on the dict
    setup(**metadata)


if __name__ == '__main__':
    do_setup()
