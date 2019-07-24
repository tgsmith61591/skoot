# -*- coding: utf-8 -*-

from skoot.utils.testing import assert_raises
from skoot.utils._docstr import _DocstrMap

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

docstr_c = """
    This is a docstr that has many sections.

    Parameters
    ----------
    x : object
        some object
        
    See Also
    --------
    SomethingElse
    
    Notes
    -----
    This should never, under ANY circumstance, be used.
"""


def assert_docstrs_equal(da, db):
    das = da.split("\n")
    dbs = db.split("\n")

    class ReportInequality(object):
        # debug class for pytest console failure repr
        def __repr__(self):
            print("Elements unequal:\n"
                  "%s\n"
                  "-------------------\n"
                  "%s" % (da, db))

    # if lengths not equal they are not the same
    debug = ReportInequality()
    assert len(das) == len(dbs), debug
    assert all(x.strip() == y.strip() for x, y in zip(das, dbs)), debug


def test_docstr_parse():
    d = _DocstrMap(docstr_a)

    # show there is no "notes" key in docstr A
    assert "Notes" not in d

    # show there is a "parameters" key in docstr A
    assert "Parameters" in d

    # show what we build corresponds to what is real
    assert_docstrs_equal(docstr_a, d.make())


def test_docstr_append_to_front():
    d = _DocstrMap(docstr_b)

    # want to append a "See also" to this
    d.add_to_front_of_section("See Also", ['    SomeThirdType'])

    expected = """
    This is another docstr. It does NOT contain a 'params'
    section, so it will not be hit be the append_params test.
    However, it does contain a 'see_also' section.

    See Also
    --------
    SomeThirdType
    SomeType
    SomeOtherType
    """

    assert_docstrs_equal(d.make(), expected)


def test_docstr_append_to_end_do_strip():
    d = _DocstrMap(docstr_b)

    # want to append a "See also" to this
    d.add_to_end_of_section("See Also", ['    SomeThirdType', ''],
                            strip_trailing_space=True)

    expected = """
    This is another docstr. It does NOT contain a 'params'
    section, so it will not be hit be the append_params test.
    However, it does contain a 'see_also' section.

    See Also
    --------
    SomeType
    SomeOtherType
    SomeThirdType
    """

    assert_docstrs_equal(d.make(), expected)


def test_docstr_append_to_end_no_strip():
    d = _DocstrMap(docstr_b)

    # want to append a "See also" to this
    d.add_to_end_of_section("See Also", ['    SomeThirdType', ''],
                            strip_trailing_space=False)

    expected = """
    This is another docstr. It does NOT contain a 'params'
    section, so it will not be hit be the append_params test.
    However, it does contain a 'see_also' section.

    See Also
    --------
    SomeType
    SomeOtherType
    
    SomeThirdType
    """

    assert_docstrs_equal(d.make(), expected)


def test_docstr_keyerror():
    # show we fail to append when key does not exist
    d = _DocstrMap(docstr_a)
    assert_raises(KeyError, d.add_to_front_of_section, "Notes", ['    Thing'])


def test_docstr_create_section():
    d = _DocstrMap(docstr_b)

    # want to append a "See also" to this
    d.create_section("New Section", ['    SomeContent', ''])

    expected = """
    This is another docstr. It does NOT contain a 'params'
    section, so it will not be hit be the append_params test.
    However, it does contain a 'see_also' section.

    See Also
    --------
    SomeType
    SomeOtherType
    
    New Section
    -----------
    SomeContent
    """

    ds = d.make()
    assert_docstrs_equal(ds, expected)

    # show create fails if not overwrite specified
    assert_raises(ValueError, d.create_section, "New Section",
                  ['    SomeOtherContent', ''], False)

    # show we CAN replace the ds section if specified
    d.create_section("New Section", ['    SomeOtherContent', ''],
                     overwrite=True)

    expected2 = """
    This is another docstr. It does NOT contain a 'params'
    section, so it will not be hit be the append_params test.
    However, it does contain a 'see_also' section.

    See Also
    --------
    SomeType
    SomeOtherType

    New Section
    -----------
    SomeOtherContent
    """

    ds = d.make()
    assert_docstrs_equal(ds, expected2)


def test_docstr_remove_section():
    d = _DocstrMap(docstr_c)

    # Remove something that DOES exist
    d.remove_section("See Also")
    expected = """
    This is a docstr that has many sections.

    Parameters
    ----------
    x : object
        some object
    
    Notes
    -----
    This should never, under ANY circumstance, be used.
    """

    assert_docstrs_equal(d.make(), expected)

    # now show we don't break down if we try to remove it again
    d.remove_section("See Also", raise_if_missing=False)
    assert_docstrs_equal(d.make(), expected)

    # show we DO break down if raise is specifed
    assert_raises(ValueError, d.remove_section, "See Also",
                  raise_if_missing=True)
