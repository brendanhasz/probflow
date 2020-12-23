import numpy as np
import pytest
import torch

from probflow.distributions import StudentT

tod = torch.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_StudentT():
    """Tests StudentT distribution"""

    # Create the distribution
    dist = StudentT()

    # Check default params
    assert dist.df == 1
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert isinstance(dist(), tod.studentT.StudentT)

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

    # Only takes Tensor-like objs
    with pytest.raises(TypeError):
        dist = StudentT(df="lalala")
    with pytest.raises(TypeError):
        dist = StudentT(loc="lalala")
    with pytest.raises(TypeError):
        dist = StudentT(scale="lalala")
    with pytest.raises(TypeError):
        dist = StudentT(df="lalala", loc="lalala", scale="lalala")

    # Should be able to set params
    dist = StudentT(df=5, loc=3, scale=2)
    assert dist.df == 5
    assert dist.loc == 3
    assert dist.scale == 2
