import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import probflow.utils.ops as O
from probflow.modules import DenseNetwork
from probflow.parameters import Parameter
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_DenseNetwork():
    """Tests probflow.modules.DenseNetwork"""

    # Should error w/ int < 1
    with pytest.raises(ValueError):
        DenseNetwork([0, 1, 5])
    with pytest.raises(ValueError):
        DenseNetwork([5, -1, 4])

    # Create the module
    dense_net = DenseNetwork([5, 4, 3, 2])

    # Test MAP outputs are same
    x = tf.random.normal([7, 5])
    samples1 = dense_net(x)
    samples2 = dense_net(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 7
    assert samples1.shape[1] == 2

    # Test samples are different
    with Sampling():
        samples1 = dense_net(x)
        samples2 = dense_net(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 7
    assert samples1.shape[1] == 2

    # parameters should return [weights, bias] for each layer
    param_list = dense_net.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 6
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in dense_net.parameters]
    assert "DenseNetwork_Dense0_weights" in param_names
    assert "DenseNetwork_Dense0_bias" in param_names
    assert "DenseNetwork_Dense1_weights" in param_names
    assert "DenseNetwork_Dense1_bias" in param_names
    assert "DenseNetwork_Dense2_weights" in param_names
    assert "DenseNetwork_Dense2_bias" in param_names
    shapes = {
        "DenseNetwork_Dense0_weights": [5, 4],
        "DenseNetwork_Dense0_bias": [1, 4],
        "DenseNetwork_Dense1_weights": [4, 3],
        "DenseNetwork_Dense1_bias": [1, 3],
        "DenseNetwork_Dense2_weights": [3, 2],
        "DenseNetwork_Dense2_bias": [1, 2],
    }
    for name, shape in shapes.items():
        param = [p for p in dense_net.parameters if p.name == name]
        assert param[0].shape == shape

    # kl_loss should return sum of KL losses
    kl_loss = dense_net.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # test Flipout
    with Sampling(flipout=True):
        samples1 = dense_net(x)
        samples2 = dense_net(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 7
    assert samples1.shape[1] == 2

    # With probabilistic = False
    dense_net = DenseNetwork([5, 4, 3, 2], probabilistic=False)
    with Sampling():
        samples1 = dense_net(x)
        samples2 = dense_net(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 7
    assert samples1.shape[1] == 2
