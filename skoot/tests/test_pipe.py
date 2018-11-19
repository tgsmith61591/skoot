# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test skoot-native objects with sklearn pipelines & grids

from __future__ import print_function

from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from skoot.base import make_transformer
from skoot.impute import SelectiveImputer
from skoot.datasets import load_iris_df
from skoot.decomposition import SelectiveTruncatedSVD, SelectivePCA
from skoot.preprocessing import (BoxCoxTransformer, SelectiveStandardScaler,
                                 SelectiveMaxAbsScaler)
from skoot.feature_selection import (MultiCorrFilter, NearZeroVarianceFilter,
                                     FeatureFilter)
from skoot.utils.testing import assert_persistable
from skoot.utils.profiling import profile_estimator

# Def data for testing
X = load_iris_df()
y = X.pop('species')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)
cv = KFold(n_splits=2, shuffle=True, random_state=42)


# this function will be made into an anonymous transformer
def subtract_k(X, k):
    return X - float(k)


# this function will be made into an anonymous transformer
def add_k(X, k):
    return X + float(k)


def test_pipeline_basic():
    pipe = Pipeline([
        ('scaler', SelectiveStandardScaler()),
        ('model', RandomForestClassifier())
    ])

    pipe.fit(X_train, y_train)


def test_pipeline_complex():
    pipe = Pipeline([
        ('scaler', SelectiveStandardScaler()),
        ('boxcox', BoxCoxTransformer(suppress_warnings=True)),
        ('pca', SelectivePCA()),
        ('svd', SelectiveTruncatedSVD()),
        ('model', RandomForestClassifier())
    ])

    pipe.fit(X_train, y_train)


def test_complex_grid_search():
    # build a pipeline
    pipe = Pipeline([
        ('dropper', FeatureFilter()),  # won't drop any
        ('collinearity', MultiCorrFilter(threshold=0.85)),
        ('imputer', SelectiveImputer()),  # pass through since all full
        ('scaler', SelectiveMaxAbsScaler()),
        ('boxcox', BoxCoxTransformer(suppress_warnings=True)),
        ('nzv', NearZeroVarianceFilter()),
        ('pca', SelectivePCA(n_components=0.9)),
        ('custom', make_transformer(subtract_k, k=1)),
        ('model', RandomForestClassifier(n_jobs=1))
    ])

    # let's define a set of hyper-parameters over which to search
    hp = {
        'collinearity__threshold': uniform(loc=.8, scale=.15),
        'collinearity__method': ['pearson', 'kendall', 'spearman'],
        'pca__n_components': uniform(loc=.75, scale=.2),
        'pca__whiten': [True, False],
        'custom__k': [1, 2, 3],
        'custom__func': [subtract_k, add_k],
        'model__n_estimators': randint(5, 10),
        'model__max_depth': randint(2, 5),
        'model__min_samples_leaf': randint(1, 5),
        'model__max_features': uniform(loc=.5, scale=.5),
        'model__max_leaf_nodes': randint(10, 15)
    }

    # define the gridsearch
    search = RandomizedSearchCV(
        pipe, hp, n_iter=2,  # just to test it even works
        scoring='accuracy', cv=cv, random_state=42,
        # in parallel so we are testing pickling of the classes
        n_jobs=2)

    # fit the search
    search.fit(X_train, y_train)

    # Show we can profile the best estimator
    profile_estimator(search.best_estimator_)

    # Assert that it's persistable
    assert_persistable(pipe, "location.pkl", X_train, y_train)
