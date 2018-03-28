import numpy as np
from numpy.testing import assert_array_equal
from skutil.preprocessing import OneHotCategoricalEncoder
import pandas as pd

# Def data for testing
X = np.array([['USA', 'RED', 'a'],
              ['MEX', 'GRN', 'b'],
              ['FRA', 'RED', 'b']])
x = pd.DataFrame.from_records(data=X, columns=['A', 'B', 'C'])
# Tack on a numeric col:
x['n'] = np.array([5, 6, 7])


def test_encode_1():
    o = OneHotCategoricalEncoder(as_df=False).fit(x)
    t = o.transform(x)

    assert_array_equal(t, np.array([
        [5., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.],
        [6., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0.],
        [7., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.]]))

    # Encode a new, unseen one
    Y = np.array([['CAN', 'BLU', 'c']])
    y = pd.DataFrame.from_records(data=Y, columns=['A', 'B', 'C'])
    y['n'] = np.array([7])
    t = o.transform(y)

    assert_array_equal(t, np.array([
        [7., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.]]))

    assert isinstance(t, np.ndarray), 'expected np.ndarray'

    # assert default is pd DF
    o = OneHotCategoricalEncoder().fit(x)
    assert isinstance(o.transform(x), pd.DataFrame)
