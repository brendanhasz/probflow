import numpy as np
import pytest

from probflow.distributions import Poisson
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Poisson():
    """Tests Poisson distribution."""
    # Create the distribution
    dist = Poisson(3)

    # Check default params
    assert dist.rate == 3

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    ppdf = lambda x, r: (
        np.power(r, x) * np.exp(-r) / np.prod(np.arange(1, x + 1))
    )
    assert np.isclose(to_numpy(dist.prob(0)), ppdf(0, 3))
    assert np.isclose(to_numpy(dist.prob(1)), ppdf(1, 3))
    assert np.isclose(to_numpy(dist.prob(2)), ppdf(2, 3))
    assert np.isclose(to_numpy(dist.prob(3)), ppdf(3, 3))
    assert np.isclose(to_numpy(dist.log_prob(0)), np.log(ppdf(0, 3)))
    assert np.isclose(to_numpy(dist.log_prob(1)), np.log(ppdf(1, 3)))
    assert np.isclose(to_numpy(dist.log_prob(2)), np.log(ppdf(2, 3)))
    assert np.isclose(to_numpy(dist.log_prob(3)), np.log(ppdf(3, 3)))
    assert to_numpy(dist.mean()) == 3

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Poisson("lalala")
