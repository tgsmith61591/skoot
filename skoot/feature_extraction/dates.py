# -*- coding: utf-8 -*-
#
# Feature engineering for dates

from __future__ import absolute_import

from ..base import BasePDTransformer
from ..utils.validation import check_dataframe

__all__ = [
    "DateFactorizer"
]


class DateFactorizer(BasePDTransformer):
    """

    """
    def __init__(self, cols=None, as_df=True, date_format=None,
                 drop_original=True):
        super(DateFactorizer, self).__init__(
            cols=cols, as_df=as_df)

        self.date_format = date_format
        self.drop_original = drop_original

    def fit(self, X, y=None):
        X, cols = check_dataframe(X, cols=self.cols, assert_all_finite=False)

        # Now the real challenge here is that some of the columns passed
        # may not be date parseable... we'll duck type it. If it fails, it
        # cannot be parsed, and we will let Pandas raise for that. No sense
        # policing it if they already do.
        subset = X[cols]

