"""
========================
Extracting date features
========================

Demonstrates how to automatically extract factor-level features from
datetime fields.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from skoot.feature_extraction import DateFactorizer
import pandas as pd
from datetime import datetime as dt

# #############################################################################
# create data
data = [
    [1, dt.strptime("06-01-2018 12:00:05", "%m-%d-%Y %H:%M:%S")],
    [2, dt.strptime("06-02-2018 13:19:12", "%m-%d-%Y %H:%M:%S")],
    [3, dt.strptime("06-03-2018 06:04:17", "%m-%d-%Y %H:%M:%S")],
    [4, dt.strptime("06-04-2018 03:56:32", "%m-%d-%Y %H:%M:%S")],
    [5, None]
]

df = pd.DataFrame.from_records(data, columns=["transaction_id", "time"])

# We can extract a multitude of features from date fields. The default will
# grab the year, month, day and hour
print("Default features:")
print(DateFactorizer(cols=['time']).fit_transform(df))

# we can specify more if we'd like:
print("\n+Minutes, +Seconds:")
print(DateFactorizer(cols=['time'],
                     features=("year", "month", "day",
                               "hour", "minute", "second")).fit_transform(df))

# And we can retain the old (pre-transform) time features if we wanted
print("\nSame as above, but retain old time column:")
print(DateFactorizer(cols=['time'],
                     drop_original=False,
                     features=("year", "month", "day",
                               "hour", "minute", "second")).fit_transform(df))
