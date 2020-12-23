import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import probflow.utils.ops as O
from probflow.data import ArrayDataGenerator, make_generator
from probflow.distributions import Bernoulli, Normal, Poisson
from probflow.models import CategoricalModel
from probflow.modules import *
from probflow.parameters import *
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_CategoricalModel(plot):
    """Tests probflow.models.CategoricalModel"""

    class MyModel(CategoricalModel):
        def __init__(self):
            self.weight = Parameter([5, 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")

        def __call__(self, x):
            return Bernoulli(x @ self.weight() + self.bias())

    # Instantiate the model
    model = MyModel()

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = 1.0 / (1.0 + np.exp(-y))
    y = np.round(y)

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=100, lr=0.1)

    # plot the predictive dist
    model.pred_dist_plot(x[:1, :])
    if plot:
        plt.title("should be one binary dist")
        plt.show()
