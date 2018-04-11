# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test skoot-native objects with sklearn pipelines & grids

from __future__ import print_function

from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from skoot.impute import SelectiveImputer
from skoot.datasets import load_iris_df
from skoot.decomposition import SelectiveTruncatedSVD, SelectivePCA
from skoot.preprocessing import BoxCoxTransformer, SelectiveScaler
from skoot.feature_selection import (MultiCorrFilter, NearZeroVarianceFilter,
                                     FeatureFilter)

# Def data for testing
X = load_iris_df()
y = X.pop('species')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)
cv = KFold(n_splits=3, shuffle=True, random_state=42)


def test_pipeline_basic():
    pipe = Pipeline([
        ('scaler', SelectiveScaler()),
        ('model', RandomForestClassifier())
    ])

    pipe.fit(X_train, y_train)


def test_pipeline_complex():
    pipe = Pipeline([
        ('scaler', SelectiveScaler()),
        ('boxcox', BoxCoxTransformer()),
        ('pca', SelectivePCA()),
        ('svd', SelectiveTruncatedSVD()),
        ('model', RandomForestClassifier())
    ])

    pipe.fit(X_train, y_train)


def test_complex_grid_search():
    # build a pipeline
    pipe = Pipeline([
        ('dropper',        FeatureFilter()),  # won't drop any
        ('collinearity',   MultiCorrFilter(threshold=0.85)),
        ('imputer',        SelectiveImputer()),  # pass through since all full
        ('scaler',         SelectiveScaler()),
        ('boxcox',         BoxCoxTransformer()),
        ('nzv',            NearZeroVarianceFilter()),
        ('pca',            SelectivePCA(n_components=0.9)),
        ('model',          RandomForestClassifier(n_jobs=1))
    ])

    # let's define a set of hyper-parameters over which to search
    hp = {
        'collinearity__threshold':    uniform(loc=.8, scale=.15),
        'collinearity__method':       ['pearson', 'kendall', 'spearman'],
        'scaler__scaler':             [None, RobustScaler()],
        'pca__n_components':          uniform(loc=.75, scale=.2),
        'pca__whiten':                [True, False],
        'model__n_estimators':        randint(5, 10),
        'model__max_depth':           randint(2, 5),
        'model__min_samples_leaf':    randint(1, 5),
        'model__max_features':        uniform(loc=.5, scale=.5),
        'model__max_leaf_nodes':      randint(10, 15)
    }

    # define the gridsearch
    search = RandomizedSearchCV(
        pipe, hp, n_iter=2,  # just to test it even works
        scoring='accuracy', cv=cv, random_state=42,
        # in parallel so we are testing pickling of the classes
        n_jobs=2)

    # fit the search
    search.fit(X_train, y_train)
