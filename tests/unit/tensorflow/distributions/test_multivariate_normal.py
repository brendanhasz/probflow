import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import MultivariateNormal

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_MultivariateNormal():
    """Tests the MultivariateNormal distribution"""

    # Create the distribution
    loc = tf.constant([1.0, 2.0])
    cov = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    dist = MultivariateNormal(loc, cov)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = MultivariateNormal("loc", cov)
    with pytest.raises(TypeError):
        dist = MultivariateNormal(loc, "cov")

    # Call should return backend obj
    assert isinstance(dist(), tfd.MultivariateNormalTriL)

    # Test methods
    prob1 = dist.prob([1.0, 2.0])
    prob2 = dist.prob([0.0, 2.0])
    prob3 = dist.prob([0.0, 3.0])
    assert prob1 > prob2
    assert prob2 > prob3
    prob1 = dist.log_prob([1.0, 2.0])
    prob2 = dist.log_prob([0.0, 2.0])
    prob3 = dist.log_prob([0.0, 3.0])
    assert prob1 > prob2
    assert prob2 > prob3

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 2
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 2
