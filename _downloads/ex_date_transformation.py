"""
================
Date transformer
================

Demonstrates how to automatically transform string date-representation
fields into datetime type fields.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.preprocessing import DateTransformer
import pandas as pd
from datetime import datetime as dt

# #############################################################################
# create data
data = [
    [1, "06/01/2018", dt.strptime("06-01-2018", "%m-%d-%Y")],
    [2, "06/02/2018", dt.strptime("06-02-2018", "%m-%d-%Y")],
    [3, "06/03/2018", dt.strptime("06-03-2018", "%m-%d-%Y")],
    [4, None, dt.strptime("06-04-2018", "%m-%d-%Y")],
    [5, "06/05/2018", None]
]

df = pd.DataFrame.from_records(data, columns=["a", "b", "c"])

# the date transformer will automatically handle existing datetime fields
# and infer the format of string datetime fields:
print("Applied to cols ['b', 'c'] with inferred format:")
print(DateTransformer(cols=['b', 'c']).fit_transform(df))

# we can also supply the format, if desired:
print("\nApplied to cols ['b', 'c'] with specified format:")
print(DateTransformer(cols=['b', 'c'],
                      date_format="%m/%d/%Y").fit_transform(df))

# Finally, if we wanted to apply the transformer to int types, we can
# add this to the permitted types
allowed = DateTransformer.DEFAULT_PERMITTED_DTYPES + ("int64",)
print("\nApplied to cols ['a', 'b', 'c'] with inferred format:")
print(DateTransformer(cols=['a', 'b', 'c'],
                      allowed_types=allowed).fit_transform(df))
