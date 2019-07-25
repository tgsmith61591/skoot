"""
=======================================
Hypothesis testing for model monitoring
=======================================

Demonstrates how to use hypothesis testing for model monitoring
and validation.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from skoot.model_validation import DistHypothesisValidator
import warnings

# #############################################################################
# Create a classification dataset and split it
seed = 42
X, y = make_classification(n_samples=5000, n_classes=2, weights=[0.1, 0.9],
                           random_state=seed, n_features=4)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=seed)

# #############################################################################
# Build a pipeline with a validator
pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                     DistHypothesisValidator(action="warn", alpha=0.05),
                     LogisticRegression(random_state=seed))

# push data through the pipeline & show we can predict on train data with
# no warnings of any kind
pipe.fit(X_train, y_train)
pipe.predict(X_train)

# #############################################################################
# Show how this can break for a dataset that does NOT conform to
# the expected distributions

X_test[:, 1] *= 5.  # multiply a feature by some large factor

# This will now produce a warning in the model validator class, since the
# two-tailed T-test will produce a P-value < alpha:
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    pipe.predict(X_test)
    print(w[0].message)

