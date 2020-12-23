import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import Deterministic
from probflow.parameters import Parameter

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Deterministic():
    """Tests Deterministic distribution"""

    # Create the distribution
    dist = Deterministic()

    # Check default params
    assert dist.loc == 0

    # Call should return backend obj
    assert isinstance(dist(), tfd.Deterministic)

    # Test methods
    assert dist.prob(0).numpy() == 1.0
    assert dist.prob(1).numpy() == 0.0
    assert dist.log_prob(0).numpy() == 0.0
    assert dist.log_prob(1).numpy() == -np.inf
    assert dist.mean().numpy() == 0.0
    assert dist.mode().numpy() == 0.0
    assert dist.cdf(-1).numpy() == 0.0
    assert dist.cdf(1).numpy() == 1.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10
    samples = dist.sample(tf.constant([10]))
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Deterministic(loc=3)
    assert dist.loc == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Deterministic(loc="lalala")

    # Test using a parameter as an argument
    p = Parameter()
    dist = Deterministic(loc=p)
    dist.sample()
