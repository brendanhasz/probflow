"""Fixtures for unit tests."""

import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import probflow as pf

# TODO: fixtures if you need them

"""
@pytest.fixture(scope="session")
def LR1_novar_unfit():
    weight = Parameter(name='thing1')
    bias = Parameter(name='thing2')
    data = Input()
    model = Normal(data*weight + bias, 1.0)
    return model
"""
