# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Utilities for wrapping scikit-learn classes, inheriting their documentation
# and amending it to fit the skoot interface.

from .iterables import ensure_iterable, is_iterable

from abc import ABCMeta
import os

__all__ = [
    'wraps_estimator'
]


_cols_doc = "    " + \
    """cols : array-like, shape=(n_features,), optional (default=None)
        The names of the columns on which to apply the transformation.
        If no column names are provided, the transformer will be fit
        on the entire frame. Note that the transformation will also
        only apply to the specified columns, and any other
        non-specified columns will still be present after
        the transformation.
        """

_as_df_doc = "    " + \
    """as_df : bool, optional (default=True)
        Whether to return a Pandas ``DataFrame`` in the ``transform``
        method. If False, will return a Numpy ``ndarray`` instead.
        Since most skoot transformers depend on explicitly-named
        ``DataFrame`` features, the ``as_df`` parameter is True by
        default.
        """

_trans_col_name_doc = "    " + \
    """trans_col_name : str, unicode or iterable, optional
        The name or list of names to apply to the transformed column(s).
        If a string is provided, it is used as a prefix for new columns.
        If an iterable is provided, its dimensions must match the number of
        produced columns. If None (default), will use the estimator class
        name as the prefix.
        """

_wrapper_msg = """
    This class wraps scikit-learn's {classname}. When a
    pd.DataFrame is passed to the ``fit`` method, the transformation is
    applied to the selected columns, which are subsequently dropped from the
    frame. All remaining columns are left alone.
    """


class _WritableDoc(ABCMeta):
    """In py27, classes inheriting from `object` do not have
    a mutable __doc__. This is shamelessly used from dask_ml

    We inherit from ABCMeta instead of type to avoid metaclass
    conflicts, since some sklearn estimators (eventually) subclass
    ABCMeta
    """
    # TODO: Py2: remove all this


class _DocstrMap(object):
    """Builds a map of the docstring sections.

    This class builds a dictionary mapping docstring section headers
    to their content. It's useful for augmenting the wrapped class
    docstr.

    Parameters
    ----------
    docstr : str or unicode
        The docstring from a class in numpydoc form.

    Attributes
    ----------
    header_ : str or unicode
        The first line of the docstr. Generally used as a header line.

    map_ : dict
        A dictionary of the docstr components keyed by (lowercase)
        section. The summary section is stored as "_section"::

        {
            "_summary": ['', '    Beginning of a summary block ',
                         '    continued on the next line'],
            "parameters": ['    ----------', 'x : object', ...],
            ...
        }

    order_ : list
        The order in which the section keys appeared
    """
    def __init__(self, docstr):
        lines = docstr.split(os.linesep)
        header, rest = lines[0], lines[1:]

        map_ = dict()
        current_section = "_summary"
        current_parts = []
        order_ = [current_section]

        # this is a bit of a hack and I'm not sure if there's a better
        # way to determine whether we're in a new section or not...
        is_underline = (lambda s: "---" in s)

        for i, e in enumerate(rest):
            v = e.lower()  # make lowercase

            # basecase: this is an underline, therefore the last part
            # should be popped out and used as a header
            if is_underline(v):
                new_section = current_parts.pop(-1)  # O(1) to pop from end
                map_[current_section] = current_parts  # map old section
                current_section = new_section.strip().lower()  # set the new
                order_.append(current_section)  # append the new key
                current_parts = [new_section]  # start new list w header prsnt

            # should happen for either case - append the str as a member
            current_parts.append(e)

        # when we make it all the way through, still have to map
        map_[current_section] = current_parts

        self.header_ = header
        self.map_ = map_
        self.order_ = order_

    def add_to_end_of_section(self, section, content,
                              strip_trailing_space=False):
        section = section.lower()  # user may not know we store as lower
        parts = self.map_[section]

        # if we are to remove the trailing space, do so now
        if strip_trailing_space:
            while not parts[-1].strip():  # truthy check on emptiness
                parts.pop(-1)
        parts += content
        self.map_[section] = parts  # should have been set in place...

    def add_to_front_of_section(self, section, content):
        section = section.lower()  # user may not know we store as lower
        sect = self.map_[section]
        # the insert point is after the header and underline (2)
        self.map_[section] = sect[:2] + content + sect[2:]

    def create_section(self, section_name, content, overwrite=False):
        key = section_name.lower()
        order = self.order_
        if key in self.map_:
            if not overwrite:
                raise ValueError("Section already exists! (%s)" % key)
            # otherwise, we are overwriting and need to pop it out
            self.remove_section(key, raise_if_missing=False)

        # if the very last slot in the order doesn't contain a newline, make
        # sure it does so we separate the new section
        last_v = self.map_[order[-1]][-1]
        if last_v.strip():  # if it's truthy (read: there's NO newline)
            self.map_[order[-1]].append('')  # add an empty str

        # add this into the order
        order.append(key)

        # now create the section
        self.map_[key] = ["    " + section_name,
                          "    " + "-" * len(section_name)] + content

    def make(self, rstrip=False):
        doc = '\n'.join([self.header_] +
                        [e for k in self.order_
                         for e in self.map_[k]])

        if rstrip:
            doc = doc.rstrip()
        return doc

    def remove_section(self, section_name, raise_if_missing=False):
        key = section_name.lower()
        popped = self.map_.pop(key, None)
        if popped is None:
            if raise_if_missing:
                raise ValueError("%s was never present in docstring sections!"
                                 % section_name)
            # this way we don't run into errors with trying to get the index
            # in a list where the element does not exist
            return

        self.order_.pop(self.order_.index(key))  # O(N)... :-|

    def __contains__(self, item):
        return item.lower() in self.map_  # O(1)


def wraps_estimator(skclass, add_sections=None, remove_sections=None):
    """Applied to classes to inherit doc from sklearn.

    Parameters
    ----------
    skclass : BaseEstimator
        The scikit-learn class from which the new estimator will inherit
        documentation. This class must have a populated docstring and a
        "Parameters" section in order to behave properly.

    add_sections : iterable or None
        Allows for a class to add new sections, or add to sections that already
        exist within the wrapped docstring. Should be in the form of a tuple::

            [("Section Name", "Section content", True),
             ("Other section name", "More content", False)]

        Where the first index is the name of the section to create/amend, the
        second is the content to add, and the third is a boolean indicating
        whether or not to overwrite the section if it exists. If False, the
        content will be appended to the section.

    remove_sections : str, unicode or iterable[str], optional (default=None)
        Any section headers that should be stripped out of the docstring. This
        is particularly useful in situations where the content of a section
        contains more than can be monkey-patched by the skoot wrapper (i.e.,
        references to scripts in the scikit-learn documentation that doesn't
        match any of the wrappers in the skoot doc).

    Notes
    -----
    This method is for internal use only. Use at your own risk!
    """
    def _copy_wrapper_doc(cls):

        # build a docstr map
        dsmap = _DocstrMap(skclass.__doc__)

        # add the insert to the header
        dsmap.header_ += " (applied to selected columns)."

        # append the wrapper message to the summary section
        insert = _wrapper_msg.format(classname=skclass.__name__)\
                             .split(os.linesep)
        dsmap.add_to_end_of_section("_summary", insert,
                                    strip_trailing_space=True)

        # update the parameters
        updated_params = os.linesep.join([
            _cols_doc, _as_df_doc, _trans_col_name_doc]).split(os.linesep)
        dsmap.add_to_front_of_section("Parameters", updated_params)

        # if we need to add sections, do so here
        if add_sections:
            for section_name, section_content, overwrite in add_sections:
                # make sure the tab exists before everything...
                section_content = ["    " + s if not s.startswith("    ")
                                   else s for s in
                                   ensure_iterable(section_content)]

                dsmap.create_section(section_name, content=section_content,
                                     overwrite=overwrite)

        # if we need to remove sections, do it here
        if remove_sections:
            # if it's not an iterable, split it on newlines
            if not is_iterable(remove_sections):
                remove_iterable = remove_sections.split(os.linesep)
            # either way, we need to ensure an iterable, even though this is
            # technically redundant... python just doesn't like assigning
            # over the name of remove_sections from inside this closure
            else:
                remove_iterable = remove_sections

            for section in remove_iterable:
                dsmap.remove_section(section, raise_if_missing=False)

        # make sure to rstrip since we may have left whitespace trailing...
        doc = dsmap.make(rstrip=True)

        # assign the docstring and the static _cls attribute, which is the
        # wrapped estimator class
        cls.__doc__ = doc
        cls._cls = skclass

        return cls

    return _copy_wrapper_doc
