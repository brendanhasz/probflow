import numpy as np
import pytest
import torch

from probflow.distributions import Cauchy

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Cauchy():
    """Tests Cauchy distribution"""

    # Create the distribution
    dist = Cauchy()

    # Check default params
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert isinstance(dist(), tod.cauchy.Cauchy)

    # Test methods
    cpdf = lambda x, m, s: 1.0 / (np.pi * s * (1 + (np.power((x - m) / s, 2))))
    assert is_close(dist.prob(0).numpy(), cpdf(0, 0, 1))
    assert is_close(dist.prob(1).numpy(), cpdf(1, 0, 1))
    assert is_close(dist.log_prob(0).numpy(), np.log(cpdf(0, 0, 1)))
    assert is_close(dist.log_prob(1).numpy(), np.log(cpdf(1, 0, 1)))
    assert dist.mean() == 0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Only works with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Cauchy(loc="lalala")
    with pytest.raises(TypeError):
        dist = Cauchy(scale="lalala")
    with pytest.raises(TypeError):
        dist = Cauchy(loc="lalala", scale="lalala")

    # Should be able to set params
    dist = Cauchy(loc=3, scale=2)
    assert dist.loc == 3
    assert dist.scale == 2
