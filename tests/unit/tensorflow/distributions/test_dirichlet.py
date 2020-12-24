import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import Dirichlet

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Dirichlet():
    """Tests Dirichlet distribution"""

    # Create the distribution
    dist = Dirichlet([1, 2, 3])

    # Check default params
    assert dist.concentration == [1, 2, 3]

    # Call should return backend obj
    assert isinstance(dist(), tfd.Dirichlet)

    # Test methods
    assert is_close(dist.prob([0, 0, 1]).numpy(), 0.0)
    assert is_close(dist.prob([0, 1, 0]).numpy(), 0.0)
    assert is_close(dist.prob([1, 0, 0]).numpy(), 0.0)
    assert is_close(dist.prob([0.3, 0.3, 0.4]).numpy(), 2.88)
    assert dist.log_prob([0, 0, 1]).numpy() == -np.inf
    assert is_close(dist.log_prob([0.3, 0.3, 0.4]).numpy(), np.log(2.88))
    assert is_close(dist.mean().numpy()[0], 1.0 / 6.0)
    assert is_close(dist.mean().numpy()[1], 2.0 / 6.0)
    assert is_close(dist.mean().numpy()[2], 3.0 / 6.0)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 3
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Dirichlet("lalala")

    # Should use the last dim if passed a Tensor arg
    dist = Dirichlet([[1, 2, 3], [3, 2, 1], [1, 1, 1], [100, 100, 100]])
    probs = dist.prob(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0.2, 0.2, 0.6],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    ).numpy()
    assert probs.ndim == 1
    assert is_close(probs[0], 0.0)
    assert is_close(probs[1], 0.0)
    assert is_close(probs[2], 2.0)
    assert probs[3] > 100.0

    # And ensure sample dims are correct
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 4
    assert samples.shape[1] == 3
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 3
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
    assert samples.shape[2] == 3
