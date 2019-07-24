# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the SMOTE balancer

from numpy.testing import assert_array_almost_equal
from sklearn.datasets import load_iris
from skoot.balance import smote_balance

import numpy as np
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

# create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
indices = [i for i in range(50)] + \
          [i for i in range(50, 60)] + \
          [i for i in range(100, 120)]
X, y = X[indices, :], y[indices]


def test_smote_simple():
    # fit with smote
    X_bal, y_bal = smote_balance(X, y, balance_ratio=1.0,
                                 random_state=42, shuffle=False)
    assert X_bal.shape[0] == 150

    # we didn't shuffle so we can assert the first 80
    # rows are all the same as they were
    assert_array_almost_equal(X_bal[:80, :], X[:80, :])

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)


def test_smote_pandas():
    X_pd = pd.DataFrame.from_records(X, columns=['a', 'b', 'c', 'd'])

    # fit with smote
    X_bal, y_bal = smote_balance(X_pd, y, balance_ratio=1.0,
                                 random_state=42, shuffle=False)
    assert X_bal.shape[0] == 150

    # we didn't shuffle so we can assert the first 80
    # rows are all the same as they were
    assert_array_almost_equal(X_bal.iloc[:80].values,
                              X_pd.iloc[:80].values)

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)
