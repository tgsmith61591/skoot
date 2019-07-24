# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the under-sampling balancer

from sklearn.datasets import load_iris
from skoot.balance import under_sample_balance
from numpy.testing import assert_array_almost_equal

import numpy as np
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

# create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
indices = [i for i in range(50)] + \
          [i for i in range(50, 60)] + \
          [i for i in range(100, 120)]

X, y = X[indices, :], y[indices]


def test_under_simple():
    # fit with under-sampling - this strips the most-populous class down to 20
    X_bal, y_bal = under_sample_balance(X, y, balance_ratio=1.0,
                                        random_state=42,
                                        shuffle=False)

    assert X_bal.shape[0] == 50

    # assert 50 of each label
    labels, counts = np.unique(y_bal, return_counts=True)
    assert counts[labels == 0][0] == 20
    assert counts[labels == 1][0] == 10
    assert counts[labels == 2][0] == 20

    # show that same class sizes won't re-sample
    X_c, y_c = iris.data, iris.target
    X_bal, y_bal = under_sample_balance(X_c, y_c, balance_ratio=1.0,
                                        random_state=42,
                                        shuffle=False)

    assert_array_almost_equal(X_bal, X_c)


def test_under_pandas():
    X_pd = pd.DataFrame.from_records(X, columns=['a', 'b', 'c', 'd'])

    # fit with under-sampling - this strips the most-populous class down to 20
    X_bal, y_bal = under_sample_balance(X_pd, y, balance_ratio=1.0,
                                        random_state=42,
                                        shuffle=False)

    assert X_bal.shape[0] == 50

    # assert 50 of each label
    labels, counts = np.unique(y_bal, return_counts=True)
    assert counts[labels == 0][0] == 20
    assert counts[labels == 1][0] == 10
    assert counts[labels == 2][0] == 20

    # assert on the output type of the balanced frame
    assert isinstance(X_bal, pd.DataFrame)
    assert X_bal is not X_pd
    assert X_bal.columns.tolist() == X_pd.columns.tolist()
