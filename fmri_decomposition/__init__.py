"""Cohort-agnostic naturalistic-viewing fMRI pipeline: stages 2 and 3.

`__version__` is stamped into every manifest by `io.write_manifest`, so it has
to exist as an attribute of the package itself -- without an `__init__.py`
this directory is only an implicit namespace package, `from . import
__version__` raises ImportError, and every extract/dfc run dies *after*
writing its shards but *before* recording what wrote them.
"""

__version__ = "0.1.0"
