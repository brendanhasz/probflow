import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import probflow.utils.ops as O
from probflow.modules import *
from probflow.parameters import *
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Dense():
    """Tests probflow.modules.Dense"""

    # Should error w/ int < 1
    with pytest.raises(ValueError):
        dense = Dense(0, 1)
    with pytest.raises(ValueError):
        dense = Dense(5, -1)

    # Create the module
    dense = Dense(5, 1)

    # Test MAP outputs are same
    x = tf.random.normal([4, 5])
    samples1 = dense(x)
    samples2 = dense(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # Test samples are different
    with Sampling():
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # parameters should return [weights, bias]
    param_list = dense.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in dense.parameters]
    assert "Dense_weights" in param_names
    assert "Dense_bias" in param_names
    weights = [p for p in dense.parameters if p.name == "Dense_weights"]
    assert weights[0].shape == [5, 1]
    bias = [p for p in dense.parameters if p.name == "Dense_bias"]
    assert bias[0].shape == [1, 1]

    # kl_loss should return sum of KL losses
    kl_loss = dense.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # test Flipout
    with Sampling(flipout=True):
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # With the probabilistic kwarg
    dense = Dense(5, 3, probabilistic=False)
    with Sampling():
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 3

    # With the weight and bias kwargs
    weight_kwargs = {"transform": tf.exp}
    bias_kwargs = {"transform": tf.math.softplus}
    dense = Dense(5, 2, weight_kwargs=weight_kwargs, bias_kwargs=bias_kwargs)
    with Sampling():
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 2
