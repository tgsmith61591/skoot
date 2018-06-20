"""
===============================
Anonymous transformers in skoot
===============================

Sometimes you have a pre-processing stage that finds itself awkwardly
positioned in the middle of your pipeline and you're left with one of
two options:

  1. Write a full transformer class
  2. Break your pipeline up into pieces

Obviously, the preferable action is the first, however many times your
function doesn't actually fit any training set parameters, so the transformer
feels like overkill.

This tutorial will introduce you to making anonymous, lightweight transformers
on the fly that will fit into your modeling pipeline seamlessly.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

# #############################################################################
# Introduce an interesting scenario
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skoot.preprocessing import SelectiveStandardScaler
from skoot.base import make_transformer
from skoot.datasets import load_iris_df

X = load_iris_df(tgt_name="target")
y = X.pop('target')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                    test_size=0.2)

# Let's say we want to scale our features with the StandardScaler, but
# for whatever reason we only want the ABSOLUTE value of the scaled values.
# We *could* create a transformer or split our pipeline, but either case is
# klunky and could interrupt our CV process in a grid search.
#
# So we'll instead define a simple commutative function that will be wrapped
# in an "anonymous" transformer
def make_abs(X):
    return X.abs()


pipe = Pipeline([
    ("scale", SelectiveStandardScaler()),
    ("abs", make_transformer(make_abs))
])

pipe.fit(X_train, y_train)
print("Absolute scaled values: ")
print(pipe.transform(X_test))
