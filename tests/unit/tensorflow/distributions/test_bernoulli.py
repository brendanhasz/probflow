import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import Bernoulli

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Bernoulli():
    """Tests Bernoulli distribution"""

    # Create the distribution
    dist = Bernoulli(0)

    # Check default params
    assert dist.logits == 0
    assert dist.probs is None

    # Call should return backend obj
    assert isinstance(dist(), tfd.Bernoulli)

    # Test methods
    assert is_close(dist.prob(0).numpy(), 0.5)
    assert is_close(dist.prob(1).numpy(), 0.5)
    assert is_close(dist.log_prob(0).numpy(), np.log(0.5))
    assert is_close(dist.log_prob(1).numpy(), np.log(0.5))
    assert dist.mean().numpy() == 0.5

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Bernoulli(probs=0.8)
    assert dist.probs == 0.8
    assert dist.logits is None
    assert is_close(dist.prob(0).numpy(), 0.2)
    assert is_close(dist.prob(1).numpy(), 0.8)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Bernoulli("lalala")
    with pytest.raises(TypeError):
        dist = Bernoulli()
