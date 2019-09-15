"""Fixtures and options for all tests."""

import pytest
import numpy as np
import tensorflow as tf


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False,
                     help="Show plots")
    #parser.addoption("--N", action="store", default=1000, type=int, 
    #                 help="Number of datapoints") #int arg
    #parser.addoption("--val_name", action="store", default="default str", 
    #                 help="description") #for a str arg


def pytest_generate_tests(metafunc):
    args = ['plot']#, 'N', 'val_name']
    for arg in args:
        val = getattr(metafunc.config.option, arg)
        if arg in metafunc.fixturenames and val is not None:
            metafunc.parametrize(arg, [val])


@pytest.fixture
def random():
    np.random.seed(12345)
    tf.random.set_seed(12345)