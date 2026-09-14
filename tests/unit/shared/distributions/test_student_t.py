import numpy as np
import pytest

from probflow.distributions import StudentT
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_StudentT():
    """Tests StudentT distribution."""
    # Create the distribution
    dist = StudentT()

    # Check default params
    assert dist.df == 1
    assert dist.loc == 0
    assert dist.scale == 1

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    cpdf = lambda x, m, s: 1.0 / (np.pi * s * (1 + (np.power((x - m) / s, 2))))
    assert np.isclose(to_numpy(dist.prob(0)), cpdf(0, 0, 1))
    assert np.isclose(to_numpy(dist.prob(1)), cpdf(1, 0, 1))
    assert np.isclose(to_numpy(dist.log_prob(0)), np.log(cpdf(0, 0, 1)))
    assert np.isclose(to_numpy(dist.log_prob(1)), np.log(cpdf(1, 0, 1)))
    assert dist.mean() == 0

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = StudentT(df=5, loc=3, scale=2)
    assert dist.df == 5
    assert dist.loc == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = StudentT(df="lalala", loc="lalala", scale="lalala")
