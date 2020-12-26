import numpy as np
import pytest
import torch
import torch.distributions as tod

from probflow.distributions import Mixture, Normal


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Mixture():
    """Tests Mixture distribution"""

    # Should fail w incorrect args
    with pytest.raises(ValueError):
        dist = Mixture(Normal(torch.tensor([1, 2]), torch.tensor([1, 2])))
    with pytest.raises(TypeError):
        dist = Mixture(Normal(torch.tensor([1, 2]), torch.tensor([1, 2])), "lala")
    with pytest.raises(TypeError):
        dist = Mixture(Normal(torch.tensor([1, 2]), torch.tensor([1, 2])), logits="lala")
    with pytest.raises(TypeError):
        dist = Mixture(Normal(torch.tensor([1, 2]), torch.tensor([1, 2])), probs="lala")
    with pytest.raises(TypeError):
        dist = Mixture("lala", probs=torch.randn([5, 3]))

    # Create the distribution
    weights = torch.randn([5, 3])
    rands = torch.randn([5, 3])
    dists = Normal(rands, torch.exp(rands))
    dist = Mixture(dists, weights)

    # Call should return backend obj
    assert isinstance(dist(), tod.MixtureSameFamily)

    # Test sampling
    samples = dist.sample()
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 1
    assert samples.shape[0] == 5
    samples = dist.sample(10)
    assert isinstance(samples, torch.Tensor)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 5

    # Test methods
    dist = Mixture(Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1e-3, 1e-3])), torch.tensor([0.5, 0.5]))
    probs = dist.prob(torch.tensor([-1.0, 1.0]))
    assert is_close(probs[0] / probs[1], 1.0)

    dist = Mixture(
        Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1e-3, 1e-3])),
        torch.tensor(np.log(np.array([0.8, 0.2]).astype("float32"))),
    )
    probs = dist.prob(torch.tensor([-1.0, 1.0]))
    assert is_close(probs[0] / probs[1], 4.0)

    dist = Mixture(
        Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1e-3, 1e-3])),
        torch.tensor(np.log(np.array([0.1, 0.9]).astype("float32"))),
    )
    probs = dist.prob(torch.tensor([-1.0, 1.0]))
    assert is_close(probs[0] / probs[1], 1.0 / 9.0)

    # try w/ weight_type
    dist = Mixture(
        Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1e-3, 1e-3])),
        logits=torch.tensor(np.log(np.array([0.1, 0.9]).astype("float32"))),
    )
    probs = dist.prob(torch.tensor([-1.0, 1.0]))
    assert is_close(probs[0] / probs[1], 1.0 / 9.0)

    dist = Mixture(
        Normal(torch.tensor([-1.0, 1.0]), torch.tensor([1e-3, 1e-3])),
        probs=torch.tensor(np.array([0.1, 0.9]).astype("float32")),
    )
    probs = dist.prob(torch.tensor([-1.0, 1.0]))
    assert is_close(probs[0] / probs[1], 1.0 / 9.0)
