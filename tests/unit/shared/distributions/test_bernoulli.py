import numpy as np
import pytest

from probflow.distributions import Bernoulli
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Bernoulli():
    """Tests Bernoulli distribution."""
    # Create the distribution
    dist = Bernoulli(0)

    # Check default params
    assert dist.logits == 0
    assert dist.probs is None

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    assert np.isclose(to_numpy(dist.prob(0)), 0.5)
    assert np.isclose(to_numpy(dist.prob(1)), 0.5)
    assert np.isclose(to_numpy(dist.log_prob(0)), np.log(0.5))
    assert np.isclose(to_numpy(dist.log_prob(1)), np.log(0.5))
    assert to_numpy(dist.mean()) == 0.5

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Bernoulli(probs=0.8)
    assert dist.probs == 0.8
    assert dist.logits is None
    assert np.isclose(to_numpy(dist.prob(0)), 0.2)
    assert np.isclose(to_numpy(dist.prob(1)), 0.8)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Bernoulli("lalala")
    with pytest.raises(TypeError):
        dist = Bernoulli()
