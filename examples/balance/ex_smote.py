"""
====================
SMOTE class balancer
====================

This example creates an imbalanced classification dataset, and applies the
`Synthetic Minority Observation TEchnique (SMOTE) <https://bit.ly/2qH3dIX>`_
class balancing technique in order to balance it by supplementing synthetic
minority class observations.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.utils.validation import check_random_state
from skoot.balance import smote_balance

plt.figure(figsize=(6, 4))

# #############################################################################
# make blobs and grossly undersample one of them
random_state = check_random_state(42)
n_samples = 1000
X, y = make_blobs(n_samples=n_samples, centers=2,
                  random_state=random_state,
                  n_features=5, cluster_std=4.5)

minority_mask = y == 0
sample_mask = ((~minority_mask) |
               (minority_mask &
                (random_state.rand(minority_mask.shape[0]) < 0.05)))

X, y = X[sample_mask, :], y[sample_mask]

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Initial imbalanced dataset")

# #############################################################################
# Balance the dataset
X_balance, y_balance = smote_balance(X, y, balance_ratio=0.2, random_state=42)

plt.subplot(122)
plt.scatter(X_balance[:, 0], X_balance[:, 1],
            c=y_balance)
plt.title("Balanced dataset")
plt.show()
