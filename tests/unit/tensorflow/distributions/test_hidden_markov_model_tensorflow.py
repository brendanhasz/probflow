import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import HiddenMarkovModel

tfd = tfp.distributions


def test_HiddenMarkovModel_tensorflow():
    """Tests hidden Markov model distribution w/ tensorflow backend."""
    initial = tf.random.normal([3])
    transition = tf.random.normal([3, 3])
    observation = tfd.Normal(
        tf.random.normal([3]), tf.exp(tf.random.normal([3]))
    )
    steps = 5
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
