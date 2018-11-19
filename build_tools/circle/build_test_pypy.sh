#!/usr/bin/env bash

set -x
set -e

# Don't test with Conda here, use virtualenv instead
pip install virtualenv

if command -v pypy3; then
    virtualenv -p $(command -v pypy3) pypy-env
elif command -v pypy; then
    virtualenv -p $(command -v pypy) pypy-env
fi

source pypy-env/bin/activate

python --version
which python

# Make sure to install scipy for Circle so we can use it as a requirement
# in the test bear pacakage
pip install --extra-index https://antocuni.github.io/pypy-wheels/ubuntu \
    numpy scipy scikit-learn pandas pytest

ccache -M 512M
export CCACHE_COMPRESS=1
export PATH=/usr/lib/ccache:$PATH
export LOKY_MAX_CPU_COUNT="2"

pip install -vv -e .

python -m pytest skoot/
