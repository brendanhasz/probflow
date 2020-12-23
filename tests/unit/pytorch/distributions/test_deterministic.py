import numpy as np
import pytest
import torch

from probflow.distributions import Deterministic
from probflow.utils.torch_distributions import get_TorchDeterministic

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_TorchDeterministic():
    """Tests the TorchDeterministic distribution"""

    TorchDeterministic = get_TorchDeterministic()

    dist = TorchDeterministic(loc=torch.tensor([2.0]), validate_args=True)

    assert is_close(dist.mean.numpy()[0], 2.0)
    assert is_close(dist.stddev, 0.0)
    assert is_close(dist.variance, 0.0)

    dist.expand([5, 2])

    dist.rsample()

    dist.log_prob(torch.tensor([1.0]))

    dist.cdf(torch.tensor([1.0]))

    dist.icdf(torch.tensor([1.0]))

    dist.entropy()


def test_Deterministic():
    """Tests Deterministic distribution"""

    # Create the distribution
    dist = Deterministic()

    # Check default params
    assert dist.loc == 0

    # Call should return backend obj
    assert isinstance(dist(), tod.distribution.Distribution)

    # Test methods
    assert dist.prob(torch.zeros([1])).numpy() == 1.0
    assert dist.prob(torch.ones([1])).numpy() == 0.0
    assert dist.log_prob(torch.zeros([1])).numpy() == 0.0
    assert dist.log_prob(torch.ones([1])).numpy() == -np.inf
    assert dist.mean().numpy() == 0.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10
    samples = dist.sample(torch.tensor([10]))
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Deterministic(loc=2)
    assert dist.loc == 2
    assert dist.prob(2 * torch.ones([1])).numpy() == 1.0
    assert dist.prob(torch.ones([1])).numpy() == 0.0

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Deterministic(loc="lalala")
