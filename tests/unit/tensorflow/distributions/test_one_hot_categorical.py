import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.distributions import OneHotCategorical

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_OneHotCategorical():
    """Tests OneHotCategorical distribution"""

    # Create the distribution
    dist = OneHotCategorical(probs=[0.1, 0.2, 0.7])

    # Check default params
    assert dist.logits is None
    assert dist.probs == [0.1, 0.2, 0.7]

    # Call should return backend obj
    assert isinstance(dist(), tfd.OneHotCategorical)

    # Test methods
    assert is_close(dist.prob([1.0, 0, 0]).numpy(), 0.1)
    assert is_close(dist.prob([0, 1.0, 0]).numpy(), 0.2)
    assert is_close(dist.prob([0, 0, 1.0]).numpy(), 0.7)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 1
    samples = dist.sample(10)
    assert isinstance(samples, tf.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3

    # Should be able to set params
    dist = OneHotCategorical(logits=[1, 7, 2])
    assert dist.logits == [1, 7, 2]
    assert dist.probs is None

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = OneHotCategorical("lalala")
    with pytest.raises(TypeError):
        dist = OneHotCategorical()

    # Multi-dim
    dist = OneHotCategorical(
        probs=[
            [0.1, 0.7, 0.2],
            [0.8, 0.1, 0.1],
            [0.01, 0.01, 0.98],
            [0.3, 0.3, 0.4],
        ]
    )
    probs = dist.prob(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert is_close(probs[0], 0.7)
    assert is_close(probs[1], 0.8)
    assert is_close(probs[2], 0.01)
    assert is_close(probs[3], 0.4)

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
