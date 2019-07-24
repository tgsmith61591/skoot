# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from numpy.testing import assert_array_almost_equal

from skoot.datasets import load_iris_df
from skoot.utils.testing import assert_transformer_asdf, assert_persistable
from skoot.decomposition import (SelectivePCA, SelectiveTruncatedSVD,
                                 SelectiveNMF, SelectiveKernelPCA,
                                 SelectiveIncrementalPCA)

# Def data for testing
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
    transformer = SelectivePCA(cols=cols, n_components=0.85,
                               trans_col_name="PC").fit(original)
    transformed = transformer.transform(original)

    # get the untouched columns to compare. These should be equal!!
    untouched_cols = transformed[comp_column_names].as_matrix()
    assert_array_almost_equal(compare_cols, untouched_cols)

    # make sure the component is present in the columns
    assert 'PC1' in transformed.columns
    assert transformed.shape[1] == 4

    # test that cols was provided
    assert isinstance(transformer.cols, list)
    assert transformer.cols[0] == cols[0]

    # show the wrapper works...
    assert transformer.n_components == 0.85
    assert not transformer.whiten  # not specified, but default value


# Test the as_df functionality
def test_selective_pca_asdf():
    assert_transformer_asdf(SelectivePCA(), X)


def test_selective_tsvd():
    original = X

    # Only perform on first two columns...
    cols = names[:2]

    comp_column_names = names[2:]
    compare_cols = original[comp_column_names].as_matrix()

    transformer = SelectiveTruncatedSVD(cols=cols, n_components=1,
                                        trans_col_name="Concept").fit(original)
    transformed = transformer.transform(original)

    untouched_cols = transformed[comp_column_names].as_matrix()
    assert_array_almost_equal(compare_cols, untouched_cols)

    assert 'Concept1' in transformed.columns
    assert transformed.shape[1] == len(comp_column_names) + 1

    # test the columns are stored appropriately
    assert isinstance(transformer.cols, list)
    assert transformer.cols == cols
    # assert transformer.cols is not cols  # No longer true after v0.20+


# Test the as_df functionality
def test_selective_tsvd_asdf():
    assert_transformer_asdf(SelectiveTruncatedSVD(), X)


def test_nmf():
    # just assert it fits/transforms
    nmf = SelectiveNMF(trans_col_name="Trans")
    trans = nmf.fit_transform(X)
    assert not X.equals(trans)
    assert "Trans1" in trans.columns


# Test the as_df functionality
def test_selective_nmf_asdf():
    assert_transformer_asdf(SelectiveNMF(), X)


def test_kpca():
    # just assert it fits/transforms
    kpca = SelectiveKernelPCA(trans_col_name="Trans")
    trans = kpca.fit_transform(X)
    assert not X.equals(trans)
    assert "Trans1" in trans.columns


# Test the as_df functionality
def test_selective_kpca_asdf():
    assert_transformer_asdf(SelectiveKernelPCA(), X)


def test_ipca():
    # just assert it fits/transforms
    ipca = SelectiveIncrementalPCA(trans_col_name="Trans")
    trans = ipca.fit_transform(X)
    assert not X.equals(trans)
    assert "Trans1" in trans.columns


# Test the as_df functionality
def test_selective_ipca_asdf():
    assert_transformer_asdf(SelectiveIncrementalPCA(), X)


def test_all_persistable():
    for est in (SelectivePCA, SelectiveTruncatedSVD,
                SelectiveNMF, SelectiveKernelPCA,
                SelectiveIncrementalPCA):
        assert_persistable(est(), location="location.pkl", X=X)
