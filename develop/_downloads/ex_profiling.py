"""
====================
Profiling estimators
====================

Skoot provides a convenient mechanism for profiling estimators and
determining whether they're performing up to standards. In this way,
you can determine the slowest point of your pipeline and diagnose
bottlenecks that could impede production performance.

Profiling can be performed on any estimator as long as the function
in which you're interested is decorated by the ``@timed_instance_method``
function decorator.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skoot.datasets import load_iris_df
from skoot.decomposition import SelectiveTruncatedSVD, SelectivePCA
from skoot.preprocessing import (BoxCoxTransformer, SelectiveStandardScaler)
from skoot.utils.profiling import profile_estimator

# #############################################################################
# load & split data
X = load_iris_df()
y = X.pop('species')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

# #############################################################################
# Fit a complex pipeline
pipe = Pipeline([
    ('scaler', SelectiveStandardScaler()),
    ('boxcox', BoxCoxTransformer(suppress_warnings=True)),
    ('pca', SelectivePCA()),
    ('svd', SelectiveTruncatedSVD()),
    ('model', RandomForestClassifier())
])

pipe.fit(X_train, y_train)

# Profile it:
print(profile_estimator(pipe))