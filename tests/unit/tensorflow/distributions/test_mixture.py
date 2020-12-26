import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import Mixture, Normal

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Mixture():
    """Tests Mixture distribution"""

    # Should fail w incorrect args
    with pytest.raises(ValueError):
        dist = Mixture(Normal([1, 2], [1, 2]))
    with pytest.raises(TypeError):
        dist = Mixture(Normal([1, 2], [1, 2]), "lala")
    with pytest.raises(TypeError):
        dist = Mixture(Normal([1, 2], [1, 2]), logits="lala")
    with pytest.raises(TypeError):
        dist = Mixture(Normal([1, 2], [1, 2]), probs="lala")
    with pytest.raises(TypeError):
        dist = Mixture("lala", probs=tf.random.normal([5, 3]))

    # Create the distribution
    weights = tf.random.normal([5, 3])
    rands = tf.random.normal([5, 3])
    dists = Normal(rands, tf.exp(rands))
    dist = Mixture(dists, weights)

    # Call should return backend obj
    assert isinstance(dist(), tfd.MixtureSameFamily)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 5
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 5

    # Test methods
    dist = Mixture(Normal([-1.0, 1.0], [1e-3, 1e-3]), [0.5, 0.5])
    probs = dist.prob([-1.0, 1.0])
    assert is_close(probs[0] / probs[1], 1.0)

    dist = Mixture(
        Normal([-1.0, 1.0], [1e-3, 1e-3]),
        np.log(np.array([0.8, 0.2]).astype("float32")),
    )
    probs = dist.prob([-1.0, 1.0])
    assert is_close(probs[0] / probs[1], 4.0)

    dist = Mixture(
        Normal([-1.0, 1.0], [1e-3, 1e-3]),
        np.log(np.array([0.1, 0.9]).astype("float32")),
    )
    probs = dist.prob([-1.0, 1.0])
    assert is_close(probs[0] / probs[1], 1.0 / 9.0)

    # try w/ weight_type
    dist = Mixture(
        Normal([-1.0, 1.0], [1e-3, 1e-3]),
        logits=np.log(np.array([0.1, 0.9]).astype("float32")),
    )
    probs = dist.prob([-1.0, 1.0])
    assert is_close(probs[0] / probs[1], 1.0 / 9.0)

    dist = Mixture(
        Normal([-1.0, 1.0], [1e-3, 1e-3]),
        probs=np.array([0.1, 0.9]).astype("float32"),
    )
    probs = dist.prob([-1.0, 1.0])
    assert is_close(probs[0] / probs[1], 1.0 / 9.0)
