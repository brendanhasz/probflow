import numpy as np
import pytest

from probflow.distributions import Dirichlet
from probflow.utils.casting import to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Dirichlet():
    """Tests Dirichlet distribution."""
    # Create the distribution
    dist = Dirichlet([1, 2, 3])

    # Check default params
    assert dist.concentration == [1, 2, 3]

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test methods
    assert np.isclose(to_numpy(dist.prob([0, 0, 1])), 0.0)
    assert np.isclose(to_numpy(dist.prob([0, 1, 0])), 0.0)
    assert np.isclose(to_numpy(dist.prob([1, 0, 0])), 0.0)
    assert np.isclose(to_numpy(dist.prob([0.3, 0.3, 0.4])), 2.88)
    assert to_numpy(dist.log_prob([0, 0, 1])) == -np.inf
    assert np.isclose(to_numpy(dist.log_prob([0.3, 0.3, 0.4])), np.log(2.88))
    assert np.isclose(to_numpy(dist.mean())[0], 1.0 / 6.0)
    assert np.isclose(to_numpy(dist.mean())[1], 2.0 / 6.0)
    assert np.isclose(to_numpy(dist.mean())[2], 3.0 / 6.0)

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 3
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 3

    # But only with Tensor-like objs
    with pytest.raises(TypeError):
        dist = Dirichlet("lalala")

    # Should use the last dim if passed a Tensor arg
    dist = Dirichlet([[1, 2, 3], [3, 2, 1], [1, 1, 1], [100, 100, 100]])
    probs = to_numpy(
        dist.prob(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0.2, 0.2, 0.6],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]
        )
    )
    assert probs.ndim == 1
    assert np.isclose(probs[0], 0.0)
    assert np.isclose(probs[1], 0.0)
    assert np.isclose(probs[2], 2.0)
    assert probs[3] > 100.0

    # And ensure sample dims are correct
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 2
    assert samples.shape[0] == 4
    assert samples.shape[1] == 3
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 3
    assert samples.shape[0] == 10
    assert samples.shape[1] == 4
    assert samples.shape[2] == 3
