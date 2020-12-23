import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import Poisson

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Poisson():
    """Tests Poisson distribution"""

    # Create the distribution
    dist = Poisson(3)

    # Check default params
    assert dist.rate == 3

    # Call should return backend obj
    assert isinstance(dist(), tfd.Poisson)

    # Test methods
    ppdf = lambda x, r: np.power(r, x) * np.exp(-r) / np.math.factorial(x)
    assert is_close(dist.prob(0).numpy(), ppdf(0, 3))
    assert is_close(dist.prob(1).numpy(), ppdf(1, 3))
    assert is_close(dist.prob(2).numpy(), ppdf(2, 3))
    assert is_close(dist.prob(3).numpy(), ppdf(3, 3))
    assert is_close(dist.log_prob(0).numpy(), np.log(ppdf(0, 3)))
    assert is_close(dist.log_prob(1).numpy(), np.log(ppdf(1, 3)))
    assert is_close(dist.log_prob(2).numpy(), np.log(ppdf(2, 3)))
    assert is_close(dist.log_prob(3).numpy(), np.log(ppdf(3, 3)))
    assert dist.mean().numpy() == 3

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Poisson("lalala")
