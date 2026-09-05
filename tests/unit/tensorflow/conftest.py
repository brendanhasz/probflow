import probflow as pf


def pytest_runtest_setup(item):
    """Provide pytest runtest setup."""
    pf.set_backend("tensorflow")
    pf.set_datatype(None)
