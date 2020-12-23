import numpy as np
import pytest
import torch

from probflow.distributions import Poisson

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Poisson():
    """Tests Poisson distribution"""

    # Create the distribution
    dist = Poisson(3)

    # Check default params
    assert dist.rate == 3

    # Call should return backend obj
    assert isinstance(dist(), tod.poisson.Poisson)

    # Test methods
    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])
    two = torch.tensor([2.0])
    three = torch.tensor([3.0])
    ppdf = lambda x, r: np.power(r, x) * np.exp(-r) / np.math.factorial(x)
    assert is_close(dist.prob(zero).numpy(), ppdf(0, 3))
    assert is_close(dist.prob(one).numpy(), ppdf(1, 3))
    assert is_close(dist.prob(two).numpy(), ppdf(2, 3))
    assert is_close(dist.prob(three).numpy(), ppdf(3, 3))
    assert is_close(dist.log_prob(zero).numpy(), np.log(ppdf(0, 3)))
    assert is_close(dist.log_prob(one).numpy(), np.log(ppdf(1, 3)))
    assert is_close(dist.log_prob(two).numpy(), np.log(ppdf(2, 3)))
    assert is_close(dist.log_prob(three).numpy(), np.log(ppdf(3, 3)))
    assert dist.mean().numpy() == 3

    # Only takes Tensor-like objs
    with pytest.raises(TypeError):
        dist = Poisson("lalala")

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10
