import numpy as np
import pytest

from probflow.distributions import InverseGamma
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_InverseGamma():
    """Tests InverseGamma distribution."""
    # Create the distribution
    dist = InverseGamma(5, 4)

    # Check default params
    assert dist.concentration == 5
    assert dist.scale == 4

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    assert np.isclose(to_numpy(dist.prob(1)), 0.78146726)
    assert np.isclose(to_numpy(dist.prob(2)), 0.09022352)
    assert np.isclose(to_numpy(dist.log_prob(1)), np.log(0.78146726))
    assert np.isclose(to_numpy(dist.log_prob(2)), np.log(0.09022352))
    assert to_numpy(dist.mean()) == 1.0

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = InverseGamma(3, 2)
    assert dist.concentration == 3
    assert dist.scale == 2

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = InverseGamma("lalala", "lalala")
