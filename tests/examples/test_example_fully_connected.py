"""Tests examples in example_fully_connected"""


import numpy as np
import tensorflow as tf

import probflow as pf


def rand(*a):
    return np.random.rand(*a).astype("float32")


def randn(*a):
    return np.random.randn(*a).astype("float32")


def zscore(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def test_example_fully_connected_manually():
    """Tests example_fully_connected#manually"""

    # Generate data
    N = 1024
    x = 10 * rand(N, 1) - 5
    y = np.sin(x) / (1 + x * x) + 0.05 * randn(N, 1)
    x = zscore(x)
    y = zscore(y)

    class DenseLayer(pf.Module):
        def __init__(self, d_in, d_out):
            self.w = pf.Parameter([d_in, d_out])
            self.b = pf.Parameter([d_out, 1])

        def __call__(self, x):
            return x @ self.w() + self.b()

    class DenseNetwork(pf.Module):
        def __init__(self, dims):
            Nl = len(dims) - 1
            self.layers = [DenseLayer(dims[i], dims[i + 1]) for i in range(Nl)]
            self.activations = [tf.nn.relu for i in range(Nl)]
            self.activations[-1] = lambda x: x

        def __call__(self, x):
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                x = self.activations[i](x)
            return x

    class DenseRegression(pf.Model):
        def __init__(self, dims):
            self.net = DenseNetwork(dims)
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(self.net(x), self.s())

    # Create and fit the model
    model = DenseRegression([5, 128, 64, 1])
    model.fit(x, y, lr=0.02, epochs=1000)


def test_example_fully_connected_modules():
    """Tests example_fully_connected using Modules"""

    # Generate data
    N = 1024
    x = 10 * rand(N, 1) - 5
    y = np.sin(x) / (1 + x * x) + 0.05 * randn(N, 1)
    x = zscore(x)
    y = zscore(y)

    class DenseRegression(pf.Model):
        def __init__(self):
            self.net = pf.Sequential(
                [
                    pf.Dense(5, 128),
                    tf.nn.relu,
                    pf.Dense(128, 64),
                    tf.nn.relu,
                    pf.Dense(64, 1),
                ]
            )
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(self.net(x), self.s())

    # Create and fit the model
    model = DenseRegression(1)
    model.fit(x, y)


def test_example_fully_connected_DenseRegression():
    """Tests example_fully_connected using DenseRegression"""

    # Generate data
    N = 1024
    D = 5
    x = randn(N, D)
    w = randn(D, 1)
    y = x @ w + 0.1 * randn(N, 1)

    # Create and fit the model
    model = pf.DenseRegression([D, 32, 32, 1])
    model.fit(x, y)


def test_example_fully_connected_DenseClassifier():
    """Tests example_fully_connected using DenseClassifier"""

    # Generate data
    N = 1024
    D = 5
    x = randn(N, D)
    w = randn(D, 1)
    y = (x @ w + 0.1 * randn(N, 1)) > 0.0

    # Create and fit the model
    model = pf.DenseClassifier([D, 32, 32, 2])
    model.fit(x, y)
