# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Test the over-sampling balancer

from sklearn.datasets import load_iris
from skoot.balance import over_sample_balance

import numpy as np
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

# create imbalance in the middle classes (0: 50, 10: 1, 20: 2)
indices = [i for i in range(50)] + \
          [i for i in range(50, 60)] + \
          [i for i in range(100, 120)]
X, y = X[indices, :], y[indices]


def test_over_simple():
    # fit with over-sampling
    X_bal, y_bal = over_sample_balance(X, y, balance_ratio=1.0,
                                       random_state=42)

    assert X_bal.shape[0] == 150

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)


def test_over_pandas():
    X_pd = pd.DataFrame.from_records(X, columns=['a', 'b', 'c', 'd'])
    X_bal, y_bal = over_sample_balance(X_pd, y, balance_ratio=1.0,
                                       random_state=42)

    assert X_bal.shape[0] == 150

    # assert 50 of each label
    _, counts = np.unique(y_bal, return_counts=True)
    assert all(c == 50 for c in counts)

    # assert on the output type of the balanced frame
    assert isinstance(X_bal, pd.DataFrame)
    assert X_bal is not X_pd
    assert X_bal.columns.tolist() == X_pd.columns.tolist()
