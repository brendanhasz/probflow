"""Fixtures for shared-backend unit tests."""

import probflow as pf


def pytest_runtest_setup(item):
    """Provide pytest runtest setup for shared tests."""
    pf.set_datatype(None)
