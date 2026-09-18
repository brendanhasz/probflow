"""Fixtures for unit tests w/ jax backend."""

import probflow as pf


def pytest_runtest_setup(item):
    """Provide pytest runtest setup."""
    pf.set_backend(pf.ProbflowBackend.JAX)
    pf.set_datatype(None)
