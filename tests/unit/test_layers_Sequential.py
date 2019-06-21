"""Tests probflow.layers.Sequential layer"""


import numpy as np
import tensorflow as tf

from probflow.layers import Add, Sub, Mul, Div, Neg, Abs, Exp, Log
from probflow.layers import Dense, Sequential
from probflow.parameters import Parameter, ScaleParameter
from probflow.distributions import Normal



def test_sequential_layer_fit():
    """Tests probflow.layers.Sequential"""

    # TODO move test to tests/integration

    # Dummy data
    x = np.random.randn(100, 4)
    w = np.random.randn(1, 4)
    b = np.random.randn()
    y = np.sum(x*w, axis=1) + b

    # Model 
    preds = Sequential(layers=[
        Dense(units=64),
        Dense(units=32),
        Dense(units=1, activation=None),
    ])
    std_dev = ScaleParameter()
    model = Normal(preds, std_dev)

    # Fit the model
    model.fit(x, y, epochs=10)

