# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function, absolute_import, division

import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal)

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import load_iris

from skoot.decomposition import SelectivePCA, SelectiveTruncatedSVD
from skoot.datasets import load_iris_df

from nose.tools import assert_raises

# Def data for testing
iris = load_iris()
names = ['a', 'b', 'c', 'd']
X = load_iris_df(include_tgt=False, names=names)


def test_selective_pca():
    # create a copy of the original
    original = X.copy()

    # set the columns we'll fit to just be the first
    cols = [names[0]]  # 'a'

    # the "other" names, and their corresponding matrix
    comp_column_names = names[1:]
    compare_cols = original[comp_column_names].as_matrix()

    # now fit PCA on the first column only
    transformer = SelectivePCA(cols=cols, n_components=0.85).fit(original)
    transformed = transformer.transform(original)

    # get the untouched columns to compare. These should be equal!!
    untouched_cols = transformed[comp_column_names].as_matrix()
    assert_array_almost_equal(compare_cols, untouched_cols)

    # make sure the component is present in the columns
    assert 'PC1' in transformed.columns
    assert transformed.shape[1] == 4
    assert isinstance(transformer.get_decomposition(), PCA)
    assert SelectivePCA().get_decomposition() is None

    # test that cols was provided
    assert isinstance(transformer.cols, list)
    assert transformer.cols[0] == cols[0]

    # what if we want to weight it?
    pca_weighted = SelectivePCA(do_weight=True, n_components=0.99, as_df=True)\
        .fit_transform(original)
    assert_raises(AssertionError, assert_array_equal, pca_weighted,
                  transformed)


# TODO:
def test_selective_tsvd():
    original = X

    # Only perform on first two columns...
    cols = names[:2]

    comp_column_names = names[2:]
    compare_cols = original[comp_column_names].as_matrix()

    transformer = SelectiveTruncatedSVD(cols=cols, n_components=1)\
        .fit(original)
    transformed = transformer.transform(original)

    untouched_cols = transformed[comp_column_names].as_matrix()
    assert_array_almost_equal(compare_cols, untouched_cols)

    assert 'Concept1' in transformed.columns
    assert transformed.shape[1] == len(comp_column_names) + 1
    assert isinstance(transformer.get_decomposition(), TruncatedSVD)
    assert SelectiveTruncatedSVD().get_decomposition() is None  # default None

    # test the columns are stored appropriately
    assert isinstance(transformer.cols, list)
    assert transformer.cols == cols
    assert transformer.cols is not cols
