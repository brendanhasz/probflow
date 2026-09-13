"""Fixtures for unit tests w/ pytorch backend."""

import probflow as pf


def pytest_runtest_setup(item):
    """Provide pytest runtest setup."""
    pf.set_backend("pytorch")
    pf.set_datatype(None)
