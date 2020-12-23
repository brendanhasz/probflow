import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import Categorical

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Categorical():
    """Tests Categorical distribution"""

    # Create the distribution
    dist = Categorical(tf.constant([0.0, 1.0, 2.0]))

    # Check default params
    assert isinstance(dist.logits, tf.Tensor)
    assert dist.probs is None

    # Call should return backend obj
    assert isinstance(dist(), tfd.Categorical)

    # Test methods
    zero = np.array([0.0])
    one = np.array([1.0])
    two = np.array([2.0])
    assert dist.prob(zero).numpy() < dist.prob(one).numpy()
    assert dist.prob(one).numpy() < dist.prob(two).numpy()
    assert dist.log_prob(zero).numpy() < dist.log_prob(one).numpy()
    assert dist.log_prob(one).numpy() < dist.log_prob(two).numpy()

    # Mean should return the mode!
    assert dist.mean().numpy() == 2

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Categorical(probs=tf.constant([0.1, 0.7, 0.2]))
    assert isinstance(dist.probs, tf.Tensor)
    assert dist.logits is None
    assert is_close(dist.prob(zero).numpy(), 0.1)
    assert is_close(dist.prob(one).numpy(), 0.7)
    assert is_close(dist.prob(two).numpy(), 0.2)
    assert dist.mean().numpy() == 1

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Categorical("lalala")
    with pytest.raises(TypeError):
        dist = Categorical()

    # Should use the last dim if passed a Tensor arg
    dist = Categorical(
        probs=tf.constant(
            [
                [0.1, 0.7, 0.2],
                [0.8, 0.1, 0.1],
                [0.01, 0.01, 0.98],
                [0.3, 0.3, 0.4],
            ]
        )
    )
    a1 = tf.constant([0.0, 1.0, 2.0, 2.0])
    a2 = tf.constant([2.0, 1.0, 0.0, 0.0])
    assert is_close(dist.prob(a1).numpy()[0], 0.1)
    assert is_close(dist.prob(a1).numpy()[1], 0.1)
    assert is_close(dist.prob(a1).numpy()[2], 0.98)
    assert is_close(dist.prob(a1).numpy()[3], 0.4)
    assert is_close(dist.prob(a2).numpy()[0], 0.2)
    assert is_close(dist.prob(a2).numpy()[1], 0.1)
    assert is_close(dist.prob(a2).numpy()[2], 0.01)
    assert is_close(dist.prob(a2).numpy()[3], 0.3)

    # And ensure sample dims are correct
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 4
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
