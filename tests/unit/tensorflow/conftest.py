"""Fixtures for unit tests w/ tensorflow backend."""

import pytest
import tensorflow as tf

import probflow as pf


def pytest_runtest_setup(item):
    pf.set_backend("tensorflow")
    pf.set_datatype(None)
