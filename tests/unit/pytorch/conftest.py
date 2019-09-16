"""Fixtures for unit tests w/ pytorch backend."""

import pytest

import probflow as pf

def pytest_runtest_setup(item):
    pf.set_backend('pytorch')
