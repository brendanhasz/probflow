"""Tests probflow.layers.Dense layer"""


import numpy as np
import tensorflow as tf

from probflow.layers import Add, Sub, Mul, Div, Neg, Abs, Exp, Log
from probflow.layers import Dense
from probflow.parameters import Parameter
from probflow.distributions import Normal



def test_dense_layer():
    """Tests probflow.layers.Dense"""

    # Float/int inputs
    l1 = Dense(units=4)
    l1._build_recursively(tf.placeholder(tf.float32, [2, 3]), [2])
    assert isinstance(l1.built_obj, tf.Tensor)
    assert l1.built_obj.shape.ndims == 2
    assert l1.built_obj.shape.dims[0].value == 2
    assert l1.built_obj.shape.dims[1].value == 4
    assert len(l1.args) == 3
    assert 'input' in l1.args
    assert 'weight' in l1.args
    assert 'bias' in l1.args

    # Reset the graph
    tf.reset_default_graph()


def test_dense_layer_fit():
    """Tests probflow.layers.Dense"""

    # TODO move test to tests/integration

    # Dummy data
    x = np.random.randn(100, 4)
    w = np.random.randn(1, 4)
    b = np.random.randn()
    y = np.sum(x*w, axis=1) + b

    # Model 
    l1 = Dense()
    model = Normal(l1, 1.0)

    # Fit the model
    model.fit(x, y, epochs=10)