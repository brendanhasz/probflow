import numpy as np
import pytest
import torch

from probflow.distributions import Bernoulli

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Bernoulli():
    """Tests Bernoulli distribution"""

    # Create the distribution
    dist = Bernoulli(0)

    # Check default params
    assert dist.logits == 0
    assert dist.probs is None

    # Call should return backend obj
    assert isinstance(dist(), tod.bernoulli.Bernoulli)

    # Test methods
    zero = torch.zeros([1])
    one = torch.ones([1])
    assert is_close(dist.prob(zero).numpy(), 0.5)
    assert is_close(dist.prob(one).numpy(), 0.5)
    assert is_close(dist.log_prob(zero).numpy(), np.log(0.5))
    assert is_close(dist.log_prob(one).numpy(), np.log(0.5))
    assert dist.mean().numpy() == 0.5

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Need to pass tensor-like probs
    with pytest.raises(TypeError):
        dist = Bernoulli("lalala")

    # Should be able to set params
    dist = Bernoulli(probs=0.8)
    assert dist.probs == 0.8
    assert dist.logits is None
    assert is_close(dist.prob(zero).numpy(), 0.2)
    assert is_close(dist.prob(one).numpy(), 0.8)
