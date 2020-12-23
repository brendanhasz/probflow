import numpy as np
import pytest
import torch

from probflow.distributions import Normal

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Normal():
    """Tests Normal distribution"""

    # Create the distribution
    dist = Normal()

    # Check default params
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert isinstance(dist(), tod.normal.Normal)

    # Test methods
    npdf = lambda x, m, s: (
        1.0
        / np.sqrt(2 * np.pi * s * s)
        * np.exp(-np.power(x - m, 2) / (2 * s * s))
    )
    assert is_close(dist.prob(0).numpy(), npdf(0, 0, 1))
    assert is_close(dist.prob(1).numpy(), npdf(1, 0, 1))
    assert is_close(dist.log_prob(0).numpy(), np.log(npdf(0, 0, 1)))
    assert is_close(dist.log_prob(1).numpy(), np.log(npdf(1, 0, 1)))
    assert dist.mean().numpy() == 0.0

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Normal(loc=3, scale=2)
    assert dist.loc == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Normal(loc="lalala", scale="lalala")
    with pytest.raises(TypeError):
        dist = Normal(loc=0, scale="lalala")
    with pytest.raises(TypeError):
        dist = Normal(loc="lalala", scale=1)
