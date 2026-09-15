import numpy as np
import pytest

from probflow.distributions import Bernoulli
from probflow.utils.casting import to_default_dtype, to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Bernoulli():
    """Tests Bernoulli distribution."""
    # Some tensors to use
    zero_tensor = to_default_dtype(0)
    one_tensor = to_default_dtype(1)

    # Create the distribution
    dist = Bernoulli(zero_tensor)

    # Check default params
    assert dist.logits == 0
    assert dist.probs is None

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    assert np.isclose(to_numpy(dist.prob(zero_tensor)), 0.5)
    assert np.isclose(to_numpy(dist.prob(one_tensor)), 0.5)
    assert np.isclose(to_numpy(dist.log_prob(zero_tensor)), np.log(0.5))
    assert np.isclose(to_numpy(dist.log_prob(one_tensor)), np.log(0.5))
    assert to_numpy(dist.mean()) == 0.5

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Bernoulli(probs=0.8)
    assert dist.probs == 0.8
    assert dist.logits is None
    assert np.isclose(to_numpy(dist.prob(zero_tensor)), 0.2)
    assert np.isclose(to_numpy(dist.prob(one_tensor)), 0.8)

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Bernoulli("lalala")
    with pytest.raises(TypeError):
        dist = Bernoulli()
