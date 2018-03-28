# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import print_function

import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from skutil.decomposition import *
from skutil.feature_selection import *
from skutil.grid_search import RandomizedSearchCV, GridSearchCV
from skutil.preprocessing import *
from skutil.utils import report_grid_score_detail
from skutil.testing import assert_fails

# Def data for testing
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)

try:
    from sklearn.model_selection import train_test_split, KFold

    # get our train/test
    X_train, X_test, y_train, y_test = train_test_split(X, iris.target, train_size=0.75, random_state=42)
    # default CV does not shuffle, so we define our own
    custom_cv = KFold(n_splits=5, shuffle=True, random_state=42)
except ImportError as i:
    from sklearn.cross_validation import train_test_split, KFold

    # get our train/test
    X_train, X_test, y_train, y_test = train_test_split(X, iris.target, train_size=0.75, random_state=42)
    custom_cv = KFold(n=y_train.shape[0], n_folds=5, shuffle=True, random_state=42)


def test_pipeline_basic():
    pipe = Pipeline([
        ('selector', FeatureRetainer(cols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'])),
        ('scaler',   SelectiveScaler()),
        ('model',    RandomForestClassifier())
    ])

    pipe.fit(X, iris.target)


def test_pipeline_complex():
    pipe = Pipeline([
        ('selector',  FeatureRetainer(cols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'])),
        ('scaler',    SelectiveScaler()),
        ('boxcox',    BoxCoxTransformer()),
        ('pca',       SelectivePCA()),
        ('svd',       SelectiveTruncatedSVD()),
        ('model',     RandomForestClassifier())
    ])

    pipe.fit(X, iris.target)


def test_random_grid():
    # build a pipeline
    pipe = Pipeline([
        ('retainer',       FeatureRetainer()),  # will retain all
        ('dropper',        FeatureDropper()),  # won't drop any
        ('mapper',         FunctionMapper()),  # pass through
        ('encoder',        OneHotCategoricalEncoder()),  # no object dtypes, so will pass through
        ('collinearity',   MulticollinearityFilterer(threshold=0.85)),
        ('imputer',        SelectiveImputer()),  # pass through
        ('scaler',         SelectiveScaler()),
        ('boxcox',         BoxCoxTransformer()),
        ('nzv',            NearZeroVarianceFilterer(threshold=1e-4)),
        ('pca',            SelectivePCA(n_components=0.9)),
        ('model',          RandomForestClassifier(n_jobs=1))
    ])

    # let's define a set of hyper-parameters over which to search
    hp = {
        'collinearity__threshold':    uniform(loc=.8, scale=.15),
        'collinearity__method':       ['pearson', 'kendall', 'spearman'],
        'scaler__scaler':             [StandardScaler(), RobustScaler()],
        'pca__n_components':          uniform(loc=.75, scale=.2),
        'pca__whiten':                [True, False],
        'model__n_estimators':        randint(5, 10),
        'model__max_depth':           randint(2, 5),
        'model__min_samples_leaf':    randint(1, 5),
        'model__max_features':        uniform(loc=.5, scale=.5),
        'model__max_leaf_nodes':      randint(10, 15)
    }

    # define the gridsearch
    search = RandomizedSearchCV(pipe, hp,
                                n_iter=2,  # just to test it even works
                                scoring='accuracy',
                                cv=2,
                                random_state=42)

    # fit the search
    search.fit(X_train, y_train)

    # test the report
    report_grid_score_detail(search, charts=False)


def test_regular_grid():
    # build a pipeline
    pipe = Pipeline([
        ('retainer',      FeatureRetainer()),  # will retain all
        ('dropper',       FeatureDropper()),  # won't drop any
        ('mapper',        FunctionMapper()),  # pass through
        ('encoder',       OneHotCategoricalEncoder()),  # no object dtypes, so will pass through
        ('collinearity',  MulticollinearityFilterer(threshold=0.85)),
        ('imputer',       SelectiveImputer()),  # pass through since no missing
        ('scaler',        SelectiveScaler()),
        ('boxcox',        BoxCoxTransformer()),
        ('nzv',           NearZeroVarianceFilterer(threshold=1e-4)),
        ('pca',           SelectivePCA(n_components=0.9)),
        ('model',         RandomForestClassifier(n_jobs=1))
    ])

    # let's define a set of hyper-parameters over which to search (exhaustively, so for the test, just do one of each)
    hp = {
        'collinearity__threshold': [0.90],
        'collinearity__method':    ['spearman'],
        'scaler__scaler':          [RobustScaler()],
        'pca__n_components':       [0.95],
        'pca__whiten':             [True],
        'model__n_estimators':     [5],
        'model__max_depth':        [5],
        'model__min_samples_leaf': [8],
        'model__max_features':     [0.75],
        'model__max_leaf_nodes':   [20]
    }

    # define the gridsearch
    search = GridSearchCV(pipe, hp,
                          scoring='accuracy',
                          cv=custom_cv,
                          verbose=1)

    # fit the search
    search.fit(X_train, y_train)
    # search.score(X_train, y_train) # throws a warning...
    search.predict(X_train)
    search.predict_proba(X_train)
    search.predict_log_proba(X_train)

    # this poses an issue.. the models are trained on X as a
    # array, and selecting the best estimator causes the retained
    # names to go away. We need to find a way to force the best_estimator_
    # to retain the names on which to start the training process.
    # search.best_estimator_.predict(X_train)

    # test the report
    report_grid_score_detail(search, charts=False)

    # test with invalid X and ys
    assert_fails(search.fit, Exception, X_train, None)

    # fit the search with a series
    search.fit(X_train, pd.Series(y_train))

    # fit the search with a DF as y
    search.fit(X_train, pd.DataFrame(pd.Series(y_train)))

    # test with invalid X and ys
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert_fails(search.fit, Exception, X_train, pd.DataFrame([pd.Series(y_train), pd.Series(y_train)]))

    # test with short y
    assert_fails(search.fit, ValueError, X_train, [0, 1, 2])
