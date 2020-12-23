import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import probflow.utils.ops as O
from probflow.modules import Dense, Sequential
from probflow.parameters import Parameter
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Sequential():
    """Tests probflow.modules.Sequential"""

    # Create the module
    seq = Sequential(
        [Dense(5, 10), tf.nn.relu, Dense(10, 3), tf.nn.relu, Dense(3, 1)]
    )

    # Steps should be list
    assert isinstance(seq.steps, list)
    assert len(seq.steps) == 5

    # Test MAP outputs are the same
    x = tf.random.normal([4, 5])
    samples1 = seq(x)
    samples2 = seq(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # Test samples are different
    with Sampling():
        samples1 = seq(x)
        samples2 = seq(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # parameters should return list of all parameters
    param_list = seq.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 6
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in seq.parameters]
    assert "Dense_weights" in param_names
    assert "Dense_bias" in param_names
    param_shapes = [p.shape for p in seq.parameters]
    assert [5, 10] in param_shapes
    assert [1, 10] in param_shapes
    assert [10, 3] in param_shapes
    assert [1, 3] in param_shapes
    assert [3, 1] in param_shapes
    assert [1, 1] in param_shapes

    # kl_loss should return sum of KL losses
    kl_loss = seq.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0
