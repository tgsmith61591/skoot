"""
=============================
Loading native skoot datasets
=============================

This example demonstrates how to use the ``datasets`` module
to load pre-bundled datasets for modeling use in skoot.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.datasets import load_adult_df

# #############################################################################
# Load the adult dataset
adult = load_adult_df(include_tgt=True, tgt_name="Salary")
print(adult.head())
