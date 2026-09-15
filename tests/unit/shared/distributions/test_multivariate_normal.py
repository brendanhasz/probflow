import pytest

from probflow.distributions import MultivariateNormal
from probflow.utils.casting import to_default_dtype
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_MultivariateNormal():
    """Tests the MultivariateNormal distribution."""
    # Create the distribution
    loc = to_default_dtype([1.0, 2.0])
    cov = to_default_dtype([[1.0, 0.0], [0.0, 1.0]])
    dist = MultivariateNormal(loc, cov)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = MultivariateNormal("loc", cov)
    with pytest.raises(TypeError):
        dist = MultivariateNormal(loc, "cov")

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    prob1 = dist.prob(to_default_dtype([1.0, 2.0]))
    prob2 = dist.prob(to_default_dtype([0.0, 2.0]))
    prob3 = dist.prob(to_default_dtype([0.0, 3.0]))
    assert prob1 > prob2
    assert prob2 > prob3
    prob1 = dist.log_prob(to_default_dtype([1.0, 2.0]))
    prob2 = dist.log_prob(to_default_dtype([0.0, 2.0]))
    prob3 = dist.log_prob(to_default_dtype([0.0, 3.0]))
    assert prob1 > prob2
    assert prob2 > prob3

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 2
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 2
