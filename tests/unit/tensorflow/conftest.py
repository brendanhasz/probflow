import probflow as pf


def pytest_runtest_setup(item):
    pf.set_backend("tensorflow")
    pf.set_datatype(None)
