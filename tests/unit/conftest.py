"""Fixtures for tests.

"""

import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *


EPOCHS = 300
N = 500


@pytest.fixture(scope="session")
def LR1_novar_unfit():
    """Single linear regression w/ no variance parameter"""
    weight = Parameter(name='thing1')
    bias = Parameter(name='thing2')
    data = Input()
    model = Normal(data*weight + bias, 1.0)
    return model


@pytest.fixture(scope="session")
def LR1_novar():
    """Single linear regression w/ no variance parameter that has been fit"""


    # Create the model
    weight = Parameter(name='LR1_novar_weight', estimator=None)
    bias = Parameter(name='LR1_novar_bias', estimator=None)
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    true_weight = 0.5
    true_bias = -1
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)

    return model


@pytest.fixture(scope="session")
def LR3_novar():
    """Multiple linear regression w/ no variance parameter that has been fit"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Create the model
    weight = Parameter(shape=Nd, name='LR3_novar_weight', estimator=None)
    bias = Parameter(name='LR3_novar_bias', estimator=None)
    data = Input()
    model = Normal(Dot(data, weight) + bias, 1.0)

    # Generate data
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)

    return model


@pytest.fixture(scope="session")
def LR3_var():
    """Multiple linear regression w/ variance parameter that has been fit
    to a large-ish dataset w/ multiple epochs"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Create the model
    weight = Parameter(shape=Nd, name='LR3_weight', estimator=None)
    bias = Parameter(name='LR3_bias', estimator=None)
    data = Input()
    std_dev = ScaleParameter(name='LR3_std_dev')
    model = Normal(Dot(data, weight) + bias, std_dev)

    # Generate data
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)

    return model