import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import InverseGamma

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_InverseGamma():
    """Tests InverseGamma distribution"""

    # Create the distribution
    dist = InverseGamma(5, 4)

    # Check default params
    assert dist.concentration == 5
    assert dist.scale == 4

    # Call should return backend obj
    assert isinstance(dist(), tfd.InverseGamma)

    # Test methods
    assert is_close(dist.prob(1).numpy(), 0.78146726)
    assert is_close(dist.prob(2).numpy(), 0.09022352)
    assert is_close(dist.log_prob(1).numpy(), np.log(0.78146726))
    assert is_close(dist.log_prob(2).numpy(), np.log(0.09022352))
    assert dist.mean().numpy() == 1.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = InverseGamma(3, 2)
    assert dist.concentration == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = InverseGamma("lalala", "lalala")
