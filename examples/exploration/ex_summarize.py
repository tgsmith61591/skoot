"""
=================
Example summarize
=================

Demonstrates how to use the ``summarize`` function to get a quick
summary of your dataset.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.exploration import summarize
from skoot.datasets import load_iris_df

# #############################################################################
# load data
iris = load_iris_df(include_tgt=True)

# add a feature of nothing but a single level of strings. This is to
# demonstrate that the summary will report on even uninformative features
iris["x5"] = "Level1"

# print the summary of the dataset
print(summarize(iris))
