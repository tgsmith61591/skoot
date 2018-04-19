# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import, division, print_function

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six

from abc import ABCMeta
import pandas as pd

from .exceptions import DeveloperError
from .utils.validation import check_dataframe, validate_test_set_columns
from .utils.iterables import is_iterable

import copy
import os

__all__ = [
    'BasePDTransformer'
]

# compat:
try:
    # PY2
    xrange
except NameError:
    xrange = range


_cols_doc = """cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be fit
        on the entire frame. Note that the transformation will also 
        only apply to the specified columns, and any other 
        non-specified columns will still be present after 
        the transformation."""

_as_df_doc = """as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead. 
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by 
        default."""

_trans_col_name_doc = """trans_col_name : str, unicode or iterable, optional
        The name or list of names to apply to the transformed column(s).
        If a string is provided, it is used as a prefix for new columns.
        If an iterable is provided, its dimensions must match the number of
        produced columns. If None (default), will use the estimator class
        name as the prefix."""

_wrapper_msg = """

    This class wraps scikit-learn's {classname}. When a pd.DataFrame is 
    passed to the ``fit`` method, the transformation is applied to the 
    selected columns, which are subsequently dropped from the frame. All 
    remaining columns are left alone."""


class BasePDTransformer(six.with_metaclass(ABCMeta, BaseEstimator,
                                           TransformerMixin)):
    __doc__ = """The base class for all Pandas frame transformers.

    Provides the base class for all skoot transformers that require
    Pandas dataframes as input.

    Parameters
    ----------
    {_cols_doc}
    
    {_as_df_doc}

    Examples
    --------
    The following is an example of how to subclass a BasePDTransformer:

        >>> from skoot.base import BasePDTransformer
        >>> class A(BasePDTransformer):
        ...     def __init__(self, cols=None, as_df=None):
        ...             super(A, self).__init__(cols, as_df)
        ...
        >>> A()
        A(as_df=None, cols=None)
    """.format(_cols_doc=_cols_doc, _as_df_doc=_as_df_doc)

    def __init__(self, cols=None, as_df=True):
        self.cols = copy.deepcopy(cols)  # do not let be mutable!
        self.as_df = as_df

    def fit(self, X, y=None):
        """Fit the transformer.

        Default behavior is not to fit any parameters and return self.
        This is useful for transformers which do not require
        parameterization, but need to fit into a pipeline.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``.
        """
        return self


class _WritableDoc(ABCMeta):
    """In py27, classes inheriting from `object` do not have
    a mutable __doc__. This is shamelessly used from dask_ml

    We inherit from ABCMeta instead of type to avoid metaclass
    conflicts, since some sklearn estimators (eventually) subclass
    ABCMeta
    """
    # TODO: Py2: remove all this


def _get_docstr_section_idx(docstr, section, case_matters=True):
    if not case_matters:
        section = section.lower()
        contains = (lambda: section in field.lower())
    else:
        contains = (lambda: section in field)

    sec_idx = -1
    for i, field in enumerate(docstr):
        if contains():
            sec_idx = i
            break
    return sec_idx


def _append_parameters(docstr):
    # search for "Parameters"
    parm_idx = _get_docstr_section_idx(docstr, "Parameters")

    # Only do this if we found "Parameters"
    if parm_idx > -1:
        # The separator is the next index
        sep_idx = parm_idx + 1

        # start one AFTER sep idx and make sure to prepend tabs
        docstr.insert(sep_idx + 1, "    " + _cols_doc + os.linesep)
        docstr.insert(sep_idx + 2, "    " + _as_df_doc + os.linesep)
        docstr.insert(sep_idx + 3, "    " + _trans_col_name_doc + os.linesep)

    return docstr


def _append_see_also(see_also, docstr, overwrite):
    if not is_iterable(see_also):
        see_also = [see_also]

    # find the "See Also" section of the docstring (sklearn devs are
    # inconsistent with the casing of "See also"...)
    sa_idx = _get_docstr_section_idx(docstr, "See Also", case_matters=False)

    # if it DOES exist, replace it. If it does NOT exist, append it.
    if sa_idx > -1:
        # we'll append to the FRONT of the existing see also section
        sep_idx = sa_idx + 1
        for see_this in see_also:
            docstr.insert(sep_idx + 1, "    " + see_this)  # no newline here
            sep_idx += 1

        # if we're here, we've added everything in and may need to remove the
        # previously existing records... so let's track where they go to/stop
        if overwrite:
            # starting from sep_idx + 1, omit until there is another
            # empty string
            omit_idcs = []
            for i in range(sep_idx + 1, len(docstr)):
                if not docstr[i]:
                    break
                else:
                    omit_idcs.append(i)
            # i don't love this... but...
            docstr = [e for i, e in enumerate(docstr)
                      if i not in set(omit_idcs)]

    # otherwise we need to create it
    else:
        return _create_new_docstr_section("See Also", docstr, see_also)
    return docstr


def _create_new_docstr_section(header, docstr, lines):
    # docstr is a LIST
    return docstr + \
           ['    ' + header, '    ' + '-' * len(header)] + \
           ['    ' + x for x in lines]


def _selective_copy_doc_for(skclass, examples=None, see_also=None,
                            overwrite_existing_see_also=False):
    """Applied to classes to inherit doc from sklearn.

    Parameters
    ----------
    skclass : BaseEstimator
        The scikit-learn class from which the new estimator will inherit
        documentation. This class must have a populated docstring and a
        "Parameters" section in order to behave properly.

    examples : str or unicode, optional (default=None)
        Any examples to inject into the documentation.

    see_also : str, unicode or iterable[str], optional (default=None)
        Any classes that should also be seen. If these are provided, any
        that may exist in sklearn will be overwritten!
    """
    def _copy_wrapper_doc(cls):
        lines = skclass.__doc__.split(os.linesep)
        header, rest = lines[0], lines[1:]

        # format msg with the classname
        insert = _wrapper_msg.format(classname=skclass.__name__)

        # Add "selective" to the header
        header += " (applied to selected columns)."

        # update the parameters
        rest = _append_parameters(rest)

        # update the See Also section
        if see_also:
            rest = _append_see_also(see_also, rest,
                                    overwrite_existing_see_also)

        # TODO: update examples

        doc = '\n'.join([header + insert] + rest)

        cls.__doc__ = doc
        return cls
    return _copy_wrapper_doc


class _SelectiveTransformerWrapper(six.with_metaclass(_WritableDoc,
                                                      BasePDTransformer)):
    # non-estimator parameters only used for the wrapper and not in set_params
    _p_names = ('cols', 'as_df', 'trans_col_name')

    # Build a selective transformer on the fly.
    #
    # This is a private method used at the head of submodules to wrap
    # sklearn modules in the selective interface. Do not include in __all__
    # since this method is for power-users/package developers. Do not
    # include in __all__ since this class is for power-users/package
    # developers.
    def __init__(self, cols=None, as_df=True, trans_col_name=None, **kwargs):

        super(_SelectiveTransformerWrapper, self).__init__(
            cols=cols, as_df=as_df)

        # this is a STATIC attribute of subclasses
        try:
            cls = self._cls
        except AttributeError:
            raise DeveloperError("_SelectiveTransformerWrapper subclasses "
                                 "must contain a static _cls attribute that "
                                 "maps to a sklearn type!")

        # get the (default) parameters for the estimator in question
        # and initialize to default
        self.estimator_ = cls()
        default_est_parms = self.estimator_.get_params(deep=True)

        # set the attributes in the estimator AND in the constructor so this
        # class behaves like sklearn in grid search
        self.estimator_.set_params(**kwargs)

        # set the kwargs here to behave like sklearn
        for k, v in six.iteritems(default_est_parms):
            if kwargs:
                v = kwargs.get(k, v)  # try get from kwargs, fail w def. value
            setattr(self, k, v)

        self.trans_col_name = trans_col_name

    def fit(self, X, y=None, **fit_kwargs):
        """Fit the wrapped transformer.

        This method will fit the wrapped sklearn transformer on the
        selected columns, leaving other columns alone.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to fit. The frame will only
            be fit on the prescribed ``cols`` (see ``__init__``) or
            all of them if ``cols`` is None. Furthermore, ``X`` will
            not be altered in the process of the fit.

        y : array-like or None, shape=(n_samples,), optional (default=None)
            Pass-through for ``sklearn.pipeline.Pipeline``. Even
            if explicitly set, will not change behavior of ``fit``.
        """
        # check on state of X and cols
        X, cols = check_dataframe(X, self.cols)

        # fit the estimator in place
        self.estimator_.fit(X[cols], **fit_kwargs)

        # the columns we fit on
        self.fit_cols_ = cols

        return self

    def transform(self, X):
        """Transform a test dataframe.

        Parameters
        ----------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The Pandas frame to transform. The operation will
            be applied to a copy of the input data, and the result
            will be returned.

        Returns
        -------
        X : pd.DataFrame, shape=(n_samples, n_features)
            The operation is applied to a copy of ``X``,
            and the result set is returned.
        """
        check_is_fitted(self, 'fit_cols_')

        # check on state of X and cols
        X, _, other_nms = check_dataframe(X, cols=self.cols,
                                          column_diff=True)

        # validate that the test set columns exist in the fit columns
        cols = self.fit_cols_
        validate_test_set_columns(cols, X.columns)

        # get the transformer
        est = self.estimator_
        transform = est.transform(X[cols])

        # get the transformed column names
        trans = self.trans_col_name
        n_trans_cols = transform.shape[1]
        if is_iterable(trans):
            if len(trans) != n_trans_cols:
                raise ValueError("dim mismatch in transformed column names "
                                 "and transformed column shape! (%i!=%i)"
                                 % (len(trans), n_trans_cols))
        # else it's some scalar
        else:
            if trans is None:  # default to class name
                trans = self.estimator_.__class__.__name__
            # this gets caught if it's None as well:
            trans = ["%s%i" % (trans, i + 1) for i in xrange(n_trans_cols)]

        # stack the transformed variables onto the RIGHT side
        right = pd.DataFrame.from_records(
            data=transform,
            columns=trans)

        # concat if needed
        x = pd.concat([X[other_nms], right], axis=1) if other_nms else right
        return x if self.as_df else x.values

    @classmethod
    def _get_param_names(cls):
        # so we can get constructor args for grid search
        # (this is a closure)
        return list(cls._p_names) + \
               cls._cls._get_param_names()  # must have _cls!
