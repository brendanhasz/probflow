import probflow as pf


def pytest_runtest_setup(item):
    """Provide pytest runtest setup."""
    # pf.set_backend(pf.ProbflowBackend.TENSORFLOW)
    pf.set_datatype(None)
