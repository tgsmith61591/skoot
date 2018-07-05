"""
================
One-hot encoding
================

Demonstrates how to use the DummyEncoder. For a more comprehensive explanation,
take a look at the
`demo on alkaline-ml.com <https://alkaline-ml.com/2018-06-18-skoot-intro>`_.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.datasets import load_adult_df
from skoot.preprocessing import DummyEncoder
from skoot.utils.dataframe import get_categorical_columns
from sklearn.model_selection import train_test_split
import pandas as pd

# #############################################################################
# load & split the data
adult = load_adult_df(tgt_name="target")
y = adult.pop("target")

# we don't want this column
_ = adult.pop("education-num")

X_train, X_test, y_train, y_test = train_test_split(adult, y, random_state=42,
                                                    test_size=0.2)

# #############################################################################
# Fit a dummy encoder
obj_cols = get_categorical_columns(X_train).columns
encoder = DummyEncoder(cols=obj_cols, handle_unknown='ignore', n_jobs=4)
encoder.fit(X_train, y_train)

# #############################################################################
# Apply to the test set
print("Test transformation:")
print(encoder.transform(X_test).head())

# #############################################################################
# Show we can work with levels we've never seen before
test_row = X_test.iloc[0]
test_row.set_value("native-country", "Atlantis")
trans = encoder.transform(pd.DataFrame([test_row]))
print("\nApplied on a row with a new native-country:")
print(trans)

nc_mask = trans.columns.str.contains("native-country")
assert trans[trans.columns[nc_mask]].sum().sum() == 0
