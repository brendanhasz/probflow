"""Tests examples in example_fully_connected"""


import numpy as np
import tensorflow as tf

from probflow import Input, Parameter, ScaleParameter
from probflow import Reshape, Relu, Matmul, Dot
from probflow import Normal


def test_fully_connected_manual():
    """Tests example in example_fully_connected/Manually"""

    # Input (D dimensions)
    D = 3
    features = Reshape(Input(), shape=[D, 1])

    # First layer
    weights1 = Parameter(shape=[128, D])
    bias1 = Parameter(shape=128)
    layer1 = Relu(Matmul(weights1, features) + bias1)

    # Second layer
    weights2 = Parameter(shape=[64, 128])
    bias2 = Parameter(shape=64)
    layer2 = Relu(Matmul(weights2, layer1) + bias2)

    # Last layer
    weights3 = Parameter(shape=64)
    bias3 = Parameter()
    predictions = Dot(weights3, Reshape(layer2)) + bias3

    # Observation distribution
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)

    # Dummy data
    N = 10
    x = np.random.randn(N, D)
    y = np.random.randn(N)
    model.fit(x, y, epochs=3)



def test_fully_connected_manual_matmul_operator():
    """Tests example in example_fully_connected/Manually w/ matmul operator"""

    # Input (D dimensions)
    D = 3
    features = Reshape(Input(), shape=[D, 1])

    # First layer
    weights1 = Parameter(shape=[128, D])
    bias1 = Parameter(shape=128)
    layer1 = Relu(weights1@features + bias1)

    # Second layer
    weights2 = Parameter(shape=[64, 128])
    bias2 = Parameter(shape=64)
    layer2 = Relu(weights2@layer1 + bias2)

    # Last layer
    weights3 = Parameter(shape=[1, 64])
    bias3 = Parameter()
    predictions = weights3@layer2 + bias3

    # Observation distribution
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)

    # Dummy data
    N = 10
    x = np.random.randn(N, D)
    y = np.random.randn(N)
    model.fit(x, y, epochs=3)
