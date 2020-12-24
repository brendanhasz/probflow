import numpy as np
import pytest
import torch

from probflow.distributions import Dirichlet

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Dirichlet():
    """Tests Dirichlet distribution"""

    # Create the distribution
    dist = Dirichlet(torch.tensor([1.0, 2.0, 3.0]))

    # Check default params
    assert isinstance(dist.concentration, torch.Tensor)

    # Call should return backend obj
    assert isinstance(dist(), tod.dirichlet.Dirichlet)

    # Test methods
    assert is_close(dist.prob(torch.tensor([0.3, 0.3, 0.4])).numpy(), 2.88)
    assert is_close(
        dist.log_prob(torch.tensor([0.3, 0.3, 0.4])).numpy(), np.log(2.88)
    )
    assert is_close(dist.mean().numpy()[0], 1.0 / 6.0)
    assert is_close(dist.mean().numpy()[1], 2.0 / 6.0)
    assert is_close(dist.mean().numpy()[2], 3.0 / 6.0)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 3
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Dirichlet("lalala")

    # Should use the last dim if passed a Tensor arg
    dist = Dirichlet(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
                [100.0, 100.0, 100.0],
            ]
        )
    )
    probs = dist.prob(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.2, 0.2, 0.6],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]
        )
    ).numpy()
    assert probs.ndim == 1
    assert is_close(probs[2], 2.0)
    assert probs[3] > 100.0

    # And ensure sample dims are correct
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 4
    assert samples.shape[1] == 3
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 3
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
    assert samples.shape[2] == 3
