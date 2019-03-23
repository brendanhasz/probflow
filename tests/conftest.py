"""Fixtures and options for all tests.

"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False, 
                     help="Show plots")
    parser.addoption("--Ndata", action="store", default=500, 
                     type=int, help="Number of datapoints")
    parser.addoption("--Nepochs", action="store", default=300, 
                     type=int, help="Number of epochs")
    #parser.addoption("--val_name", action="store", default="default str", 
    #                 help="description") #for a str arg


def pytest_generate_tests(metafunc):
    args = ['plot', 'Ndata', 'Nepochs']
    for arg in args:
        val = getattr(metafunc.config.option, arg)
        if arg in metafunc.fixturenames and val is not None:
            metafunc.parametrize(arg, [val])
