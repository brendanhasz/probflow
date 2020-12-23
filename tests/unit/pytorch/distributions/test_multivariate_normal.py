import numpy as np
import pytest
import torch

from probflow.distributions import MultivariateNormal

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_MultivariateNormal():
    """Tests the MultivariateNormal distribution"""

    # Create the distribution
    loc = torch.Tensor([1.0, 2.0])
    cov = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    dist = MultivariateNormal(loc, cov)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = MultivariateNormal("loc", cov)
    with pytest.raises(TypeError):
        dist = MultivariateNormal(loc, "cov")

    # Call should return backend obj
    assert isinstance(dist(), tod.multivariate_normal.MultivariateNormal)

    # Test methods
    prob1 = dist.prob(torch.Tensor([1.0, 2.0]))
    prob2 = dist.prob(torch.Tensor([0.0, 2.0]))
    prob3 = dist.prob(torch.Tensor([0.0, 3.0]))
    assert prob1 > prob2
    assert prob2 > prob3
    prob1 = dist.log_prob(torch.Tensor([1.0, 2.0]))
    prob2 = dist.log_prob(torch.Tensor([0.0, 2.0]))
    prob3 = dist.log_prob(torch.Tensor([0.0, 3.0]))
    assert prob1 > prob2
    assert prob2 > prob3

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 2
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 2
