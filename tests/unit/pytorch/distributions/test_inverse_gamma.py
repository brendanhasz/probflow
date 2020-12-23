import numpy as np
import pytest
import torch

from probflow.distributions import InverseGamma

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_InverseGamma():
    """Tests InverseGamma distribution"""

    # Create the distribution
    dist = InverseGamma(5, 4)

    # Check default params
    assert dist.concentration == 5
    assert dist.scale == 4

    # Call should return backend obj
    assert isinstance(
        dist(), tod.transformed_distribution.TransformedDistribution
    )

    # Test methods
    one = torch.ones([1])
    two = 2.0 * torch.ones([1])
    assert is_close(dist.prob(one).numpy(), 0.78146726)
    assert is_close(dist.prob(two).numpy(), 0.09022352)
    assert is_close(dist.log_prob(one).numpy(), np.log(0.78146726))
    assert is_close(dist.log_prob(two).numpy(), np.log(0.09022352))
    # assert dist.mean().numpy() == 1.0 #NOTE: pytorch doesn't implement mean()

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = InverseGamma(3, 2)
    assert dist.concentration == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = InverseGamma(5, "lalala")
    with pytest.raises(TypeError):
        dist = InverseGamma("lalala", 4)
    with pytest.raises(TypeError):
        dist = InverseGamma("lalala", "lalala")
