import numpy as np
import pytest

from probflow.distributions import Categorical
from probflow.utils.casting import to_default_dtype, to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Categorical():
    """Tests Categorical distribution."""
    # Create the distribution
    dist = Categorical(to_default_dtype([0.0, 1.0, 2.0]))

    # Check default params
    assert is_backend_tensor(dist.logits)
    assert dist.probs is None

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    zero = np.array([0.0])
    one = np.array([1.0])
    two = np.array([2.0])
    assert to_numpy(dist.prob(zero)) < to_numpy(dist.prob(one))
    assert to_numpy(dist.prob(one)) < to_numpy(dist.prob(two))
    assert to_numpy(dist.log_prob(zero)) < to_numpy(dist.log_prob(one))
    assert to_numpy(dist.log_prob(one)) < to_numpy(dist.log_prob(two))

    # Mean should return the mode!
    assert to_numpy(dist.mean()) == 2

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 0
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 10

    # Should be able to set params
    dist = Categorical(probs=to_default_dtype([0.1, 0.7, 0.2]))
    assert is_backend_tensor(dist.probs)
    assert dist.logits is None
    assert np.isclose(to_numpy(dist.prob(zero)), 0.1)
    assert np.isclose(to_numpy(dist.prob(one)), 0.7)
    assert np.isclose(to_numpy(dist.prob(two)), 0.2)
    assert to_numpy(dist.mean()) == 1

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Categorical("lalala")
    with pytest.raises(TypeError):
        dist = Categorical()

    # Should use the last dim if passed a Tensor arg
    dist = Categorical(
        probs=to_default_dtype(
            [
                [0.1, 0.7, 0.2],
                [0.8, 0.1, 0.1],
                [0.01, 0.01, 0.98],
                [0.3, 0.3, 0.4],
            ]
        )
    )
    a1 = to_default_dtype([0.0, 1.0, 2.0, 2.0])
    a2 = to_default_dtype([2.0, 1.0, 0.0, 0.0])
    assert np.isclose(to_numpy(dist.prob(a1))[0], 0.1)
    assert np.isclose(to_numpy(dist.prob(a1))[1], 0.1)
    assert np.isclose(to_numpy(dist.prob(a1))[2], 0.98)
    assert np.isclose(to_numpy(dist.prob(a1))[3], 0.4)
    assert np.isclose(to_numpy(dist.prob(a2))[0], 0.2)
    assert np.isclose(to_numpy(dist.prob(a2))[1], 0.1)
    assert np.isclose(to_numpy(dist.prob(a2))[2], 0.01)
    assert np.isclose(to_numpy(dist.prob(a2))[3], 0.3)

    # And ensure sample dims are correct
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 4
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
