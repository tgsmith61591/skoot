"""
=============================
Getting acquainted with skoot
=============================

This example walks through the package layout and where various
transformers/classes can be located, as well as displays some nuances
between scikit-learn and skoot.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

# #############################################################################
# Skoot is laid out much like scikit-learn. That is, many of the same modules
# exist in skoot that are present in scikit. For example:
from skoot import decomposition
print(dir(decomposition))  # many are similar to sklearn classes
print("")

# #############################################################################
# Skoot also has a dataset interface, like sklearn. Except it returns
# dataframes rather than numpy arrays:
from skoot.datasets import load_iris_df
df = load_iris_df(include_tgt=True, tgt_name='Species')
print(df.head())
print("")

# #############################################################################
# All skoot transformers are based on the BasePDTransformer:
from skoot.base import BasePDTransformer

print(BasePDTransformer.__doc__)
