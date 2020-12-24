import numpy as np
import pytest
import torch

from probflow.distributions import Gamma

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Gamma():
    """Tests Gamma distribution"""

    # Create the distribution
    dist = Gamma(5, 4)

    # Check default params
    assert dist.concentration == 5
    assert dist.rate == 4

    # Call should return backend obj
    assert isinstance(dist(), tod.gamma.Gamma)

    # Test methods
    zero = torch.zeros([1])
    one = torch.ones([1])
    assert is_close(dist.prob(zero).numpy(), 0.0)
    assert is_close(dist.prob(one).numpy(), 0.78146726)
    assert dist.log_prob(zero).numpy() == -np.inf
    assert is_close(dist.log_prob(one).numpy(), np.log(0.78146726))
    assert is_close(dist.mean(), 5.0 / 4.0)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Gamma(3, 2)
    assert dist.concentration == 3
    assert dist.rate == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Gamma(5, "lalala")
    with pytest.raises(TypeError):
        dist = Gamma("lalala", 4)
    with pytest.raises(TypeError):
        dist = Gamma("lalala", "lalala")
