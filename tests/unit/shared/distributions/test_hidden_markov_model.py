import numpy as np
import pytest

import probflow.utils.ops as O
from probflow.distributions import HiddenMarkovModel, Normal
from probflow.utils.settings import ProbflowBackend, get_backend
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_HiddenMarkovModel():
    """Tests hidden Markov model distribution."""
    # Create the distribution (3 states)
    initial = O.randn([3])
    transition = O.randn([3, 3])
    observation = Normal(O.randn([3]), O.exp(O.randn([3])))
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
    assert is_backend_distribution(dist())

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 5
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
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
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch
        import torch.distributions as dist_module

        observation = dist_module.Normal(
            torch.randn([3]), torch.exp(torch.randn([3]))
        )
    else:
        import tensorflow as tf
        import tensorflow_probability as tfp

        tfd = tfp.distributions
        observation = tfd.Normal(
            tf.random.normal([3]), tf.exp(tf.random.normal([3]))
        )
    dist = HiddenMarkovModel(initial, transition, observation, steps)

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 5
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 5
