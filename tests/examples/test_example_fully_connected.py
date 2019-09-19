"""Tests examples in example_fully_connected"""



import numpy as np
import tensorflow as tf

import probflow as pf



def test_example_fully_connected_manually():
    """Tests example_fully_connected#manually"""

    # TODO: generate data

    class DenseLayer(pf.Module):

        def __init__(self, d_in, d_out):
            self.w = pf.Parameter([d_in, d_out])
            self.b = pf.Parameter([d_out, 1])

        def __call__(self, x):
            return x @ self.w() + self.b()


    class DenseNetwork(pf.Module):
        
        def __init__(self, dims):
            Nl = len(dims)-1
            self.layers = [DenseLayer(dims[i], dims[i+1]) for i in range(Nl)]
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
    model.fit(x, y)



def test_example_fully_connected_modules():
    """Tests example_fully_connected using Modules"""

    # TODO: generate data

    class DenseRegression(pf.Model):
        
        def __init__(self):
            self.net = pf.Sequential([
                pf.Dense(5, 128),
                tf.nn.relu,
                pf.Dense(128, 64),
                tf.nn.relu,
                pf.Dense(64, 1),
            ])
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(self.net(x), self.s())

    # Create and fit the model
    model = DenseRegression([5, 128, 64, 1])
    model.fit(x, y)



def test_example_fully_connected_DenseRegression():
    """Tests example_fully_connected using DenseRegression"""

    # TODO: generate data

    # Create and fit the model
    model = pf.DenseRegression([5, 128, 64, 1])
    model.fit(x, y)



def test_example_fully_connected_DenseClassifier():
    """Tests example_fully_connected using DenseClassifier"""

    # TODO: generate data

    # Create and fit the model
    model = pf.DenseClassifier([5, 128, 64, 1])
    model.fit(x, y)
