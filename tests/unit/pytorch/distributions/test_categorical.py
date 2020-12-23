import numpy as np
import pytest
import torch

from probflow.distributions import Categorical

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Categorical():
    """Tests Categorical distribution"""

    # Create the distribution
    dist = Categorical(torch.tensor([0.0, 1.0, 2.0]))

    # Check default params
    assert isinstance(dist.logits, torch.Tensor)
    assert dist.probs is None

    # Call should return backend obj
    assert isinstance(dist(), tod.categorical.Categorical)

    # Test methods
    zero = torch.zeros([1])
    one = torch.ones([1])
    two = 2.0 * torch.ones([1])
    assert dist.prob(zero).numpy() < dist.prob(one).numpy()
    assert dist.prob(one).numpy() < dist.prob(two).numpy()
    assert dist.log_prob(zero).numpy() < dist.log_prob(one).numpy()
    assert dist.log_prob(one).numpy() < dist.log_prob(two).numpy()

    """
    # Mean should return the mode!
    assert dist.mean().numpy() == 2
    #NOTE: pytorch doesn't implement mean()
    """

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Categorical(probs=torch.tensor([0.1, 0.7, 0.2]))
    assert isinstance(dist.probs, torch.Tensor)
    assert dist.logits is None
    assert is_close(dist.prob(zero).numpy(), 0.1)
    assert is_close(dist.prob(one).numpy(), 0.7)
    assert is_close(dist.prob(two).numpy(), 0.2)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Categorical("lalala")

    # Should use the last dim if passed a Tensor arg
    dist = Categorical(
        probs=torch.tensor(
            [
                [0.1, 0.7, 0.2],
                [0.8, 0.1, 0.1],
                [0.01, 0.01, 0.98],
                [0.3, 0.3, 0.4],
            ]
        )
    )
    v1 = torch.tensor([0, 1, 2, 2])
    v2 = torch.tensor([2, 1, 0, 0])
    assert is_close(dist.prob(v1).numpy()[0], 0.1)
    assert is_close(dist.prob(v1).numpy()[1], 0.1)
    assert is_close(dist.prob(v1).numpy()[2], 0.98)
    assert is_close(dist.prob(v1).numpy()[3], 0.4)
    assert is_close(dist.prob(v2).numpy()[0], 0.2)
    assert is_close(dist.prob(v2).numpy()[1], 0.1)
    assert is_close(dist.prob(v2).numpy()[2], 0.01)
    assert is_close(dist.prob(v2).numpy()[3], 0.3)

    # And ensure sample dims are correct
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 4
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
