import numpy as np
import pytest

from probflow.distributions import Normal
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Normal():
    """Tests Normal distribution."""
    # Create the distribution
    dist = Normal()

    # Check default params
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    npdf = lambda x, m, s: (
        1.0
        / np.sqrt(2 * np.pi * s * s)
        * np.exp(-np.power(x - m, 2) / (2 * s * s))
    )
    assert np.isclose(to_numpy(dist.prob(0)), npdf(0, 0, 1))
    assert np.isclose(to_numpy(dist.prob(1)), npdf(1, 0, 1))
    assert np.isclose(to_numpy(dist.log_prob(0)), np.log(npdf(0, 0, 1)))
    assert np.isclose(to_numpy(dist.log_prob(1)), np.log(npdf(1, 0, 1)))
    assert to_numpy(dist.mean()) == 0.0

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Normal(loc=3, scale=2)
    assert dist.loc == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Normal(loc="lalala", scale="lalala")
