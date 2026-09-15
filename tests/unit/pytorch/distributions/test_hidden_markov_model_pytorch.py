import torch

from probflow.distributions import HiddenMarkovModel


def test_HiddenMarkovModel_pytorch():
    """Tests hidden Markov model distribution w/ pytorch backend."""
    initial = torch.randn([3])
    transition = torch.randn([3, 3])
    observation = torch.distributions.Normal(
        torch.randn([3]), torch.exp(torch.randn([3]))
    )
    steps = 5
    dist = HiddenMarkovModel(initial, transition, observation, steps)

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
