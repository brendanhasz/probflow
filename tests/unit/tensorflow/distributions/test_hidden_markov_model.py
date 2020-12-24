import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import HiddenMarkovModel, Normal

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_HiddenMarkovModel():
    """Tests hidden Markov model distribution"""

    # Create the distribution (3 states)
    initial = tf.random.normal([3])
    transition = tf.random.normal([3, 3])
    observation = Normal(tf.random.normal([3]), tf.exp(tf.random.normal([3])))
    steps = 5
    dist = HiddenMarkovModel(initial, transition, observation, steps)

    # Should fail w incorrect args
    with pytest.raises(TypeError):
        HiddenMarkovModel("lala", transition, observation, steps)
    with pytest.raises(TypeError):
        HiddenMarkovModel(initial, "lala", observation, steps)
    with pytest.raises(TypeError):
        HiddenMarkovModel(initial, transition, observation, "lala")
    with pytest.raises(ValueError):
        HiddenMarkovModel(initial, transition, observation, -1)

    # Call should return backend obj
    assert isinstance(dist(), tfd.HiddenMarkovModel)

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
    probs = dist.prob([-1.0, 1.0, 0.0, 0.0, 0.0])
    assert probs.ndim == 0
    probs = dist.prob(np.random.randn(7, 5))
    assert probs.ndim == 1
    assert probs.shape[0] == 7

    # Should also work w/ a backend distribution
    observation = tfd.Normal(
        tf.random.normal([3]), tf.exp(tf.random.normal([3]))
    )
    dist = HiddenMarkovModel(initial, transition, observation, steps)

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
