import numpy as np
import pytest

from probflow.distributions import Deterministic
from probflow.parameters import Parameter
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Deterministic():
    """Tests Deterministic distribution."""
    # Create the distribution
    dist = Deterministic()

    # Check default params
    assert dist.loc == 0

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    assert to_numpy(dist.prob(0)) == 1.0
    assert to_numpy(dist.prob(1)) == 0.0
    assert to_numpy(dist.log_prob(0)) == 0.0
    assert to_numpy(dist.log_prob(1)) == -np.inf
    assert to_numpy(dist.mean()) == 0.0
    assert to_numpy(dist.mode()) == 0.0
    assert to_numpy(dist.cdf(-1)) == 0.0
    assert to_numpy(dist.cdf(1)) == 1.0

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10
    samples = dist.sample(np.array([10]))
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Deterministic(loc=3)
    assert dist.loc == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Deterministic(loc="lalala")

    # Test using a parameter as an argument
    p = Parameter()
    dist = Deterministic(loc=p)
    dist.sample()
