"""Tests examples in example_linear_regression"""


import numpy as np
import pandas as pd
import tensorflow as tf

import probflow as pf


def test_example_linear_regression_simple():
    """Tests example_linear_regression simple linear regression"""

    # Data
    randn = lambda *a: np.random.randn(*a).astype("float32")
    x = randn(100)
    y = 2 * x - 1 + randn(100)

    class SimpleLinearRegression(pf.Model):
        def __init__(self):
            self.w = pf.Parameter()
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(x * self.w() + self.b(), self.s())

    # Create and fit the model
    model = SimpleLinearRegression()
    model.fit(x, y)

    # TODO: inspection code


def test_example_linear_regression_multiple():
    """Tests example_linear_regression multiple linear regression"""

    # Data
    D = 3
    N = 256
    randn = lambda *a: np.random.randn(*a).astype("float32")
    x = pd.DataFrame(randn(N, D))
    w = randn(D, 1)
    y = x @ w - 1 + 0.2 * randn(N, 1)

    class MultipleLinearRegression(pf.Model):
        def __init__(self, dims):
            self.w = pf.Parameter([dims, 1])
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(x @ self.w() + self.b(), self.s())

    # Create and fit the model
    model = MultipleLinearRegression(D)
    model.fit(x, y, lr=0.1, epochs=300)

    # TODO: inspection code
