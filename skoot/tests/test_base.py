# -*- coding: utf-8 -*-

from __future__ import absolute_import

from skoot.base import (_append_see_also, _append_parameters, _cols_doc,
                        _as_df_doc, _trans_col_name_doc)


docstr_a = """
    This is a docstr. It does NOT contain a 'see_also'
    section, so it will not be hit directly be the see_also test.
    However, it does contain a 'params' section.
    
    Parameters
    ----------
    x : object
        some object
"""


docstr_b = """
    This is another docstr. It does NOT contain a 'params'
    section, so it will not be hit be the append_params test.
    However, it does contain a 'see_also' section.
    
    See Also
    --------
    SomeType
    SomeOtherType
"""


# Add an element to the FRONT of See Also
def test_append_see_also():
    rest = docstr_b.split("\n")
    amended = "\n".join(_append_see_also(["First Thing"], rest, False))

    expected = """
    See Also
    --------
    First Thing
    SomeType
    SomeOtherType"""

    assert expected in amended


# Add an element to the FRONT of See Also
def test_append_see_also_w_omit():
    rest = docstr_b.split("\n")
    amended = "\n".join(_append_see_also(["First Thing"], rest, True))

    expected = """
    See Also
    --------
    First Thing\n"""  # we get an extra new line here... hmm

    assert expected in amended


# Add a See Also section to a docstr without one (with an iterable)
def test_add_see_also_iterable():
    rest = docstr_a.split("\n")
    amended = "\n".join(_append_see_also(["a", "b", "c"], rest, False))

    expected = """
    See Also
    --------
    a
    b
    c"""

    assert expected in amended


# Add a See Also section to a docstr without one (with a string)
def test_add_see_also_str():
    rest = docstr_a.split("\n")
    amended = "\n".join(_append_see_also("Some Thing", rest, False))

    expected = """
    See Also
    --------
    Some Thing"""

    assert expected in amended


# There will be three "Parameters" added here
def test_append_params():
    rest = docstr_a.split("\n")
    amended = "\n".join(_append_parameters(rest))
    assert _cols_doc in amended
    assert _as_df_doc in amended
    assert _trans_col_name_doc in amended

    # also show the other parameter stuck around...
    assert "x : object" in amended


# There will not be a "Parameters" section here
def test_no_append_params():
    rest = docstr_b.split("\n")
    amended = "\n".join(_append_parameters(rest))
    assert not any(x in amended
                   for x in (_cols_doc, _as_df_doc,
                             _trans_col_name_doc))
