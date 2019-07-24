"""
==============================
Undersampling minority samples
==============================

This example creates an imbalanced classification dataset, and
under-samples it to balance the class ratios. Under-sampling is
typically only possible if your dataset is large enough to
facilitate losing some samples from the majority class. If not,
you may later suffer from a high variance problem.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.datasets import make_classification
from skoot.balance import under_sample_balance
import pandas as pd

# #############################################################################
# Create an imbalanced dataset
X, y = make_classification(n_samples=5000, n_classes=2, weights=[0.1, 0.9],
                           random_state=42)

# get counts:
zero_mask = y == 0
print("Num zero class (pre-balance): %i" % zero_mask.sum())
print("Num one class (pre-balance): %i\n" % (~zero_mask).sum())

# #############################################################################
# Balance the dataset
X_balance, y_balance = under_sample_balance(X, y, balance_ratio=0.2,
                                            random_state=42)

# get the new counts
new_mask = y_balance == 0
print("Num zero class (post-balance): %i" % new_mask.sum())
print("Num one class (post-balance): %i" % (~new_mask).sum())
print("Num samples (post-balance): %i" % X_balance.shape[0])

# #############################################################################
# This also works for pandas DataFrames

X_balance_df, _ = under_sample_balance(pd.DataFrame.from_records(X),
                                       y, balance_ratio=0.2,
                                       random_state=42)

print(X_balance_df.head())
