"""Tests examples in example_fully_connected"""



import numpy as np
import tensorflow as tf

import probflow as pf



def test_example_linear_regression_simple():
    """Tests example_linear_regression simple linear regression"""

    # TODO: generate data

    class SimpleLinearRegression(pf.Model):

        def __init__(self):
            self.w = pf.Parameter()
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(x*self.w()+self.b(), self.s())

    # Create and fit the model
    model = SimpleLinearRegression()
    model.fit(x, y)

    # TODO: inspection code



def test_example_linear_regression_multiple():
    """Tests example_linear_regression multiple linear regression"""

    # TODO: generate data

    class MultipleLinearRegression(pf.Model):

        def __init__(self, dims):
            self.w = pf.Parameter([dims, 1])
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(x @ self.w() + self.b(), self.s())

    # Create and fit the model
    model = MultipleLinearRegression()
    model.fit(x, y)

    # TODO: inspection code


def test_example_linear_regression_multiple():
    """Tests example_linear_regression multiple linear regression"""

    # TODO: generate data

    # Create and fit the model
    model = pf.LinearRegression()
    model.fit(x, y)
