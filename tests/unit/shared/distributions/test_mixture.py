import numpy as np
import pytest

import probflow.utils.ops as O
from probflow.distributions import Mixture, Normal
from probflow.utils.casting import to_default_dtype
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Mixture():
    """Tests Mixture distribution."""
    # Should fail w incorrect args
    with pytest.raises(ValueError):
        dist = Mixture(Normal([1, 2], [1, 2]))
    with pytest.raises(TypeError):
        dist = Mixture(Normal([1, 2], [1, 2]), "lala")
    with pytest.raises(TypeError):
        dist = Mixture(Normal([1, 2], [1, 2]), logits="lala")
    with pytest.raises(TypeError):
        dist = Mixture(Normal([1, 2], [1, 2]), probs="lala")
    with pytest.raises(TypeError):
        dist = Mixture("lala", probs=O.randn([5, 3]))

    # Create the distribution
    weights = O.randn([5, 3])
    rands = O.randn([5, 3])
    dists = Normal(rands, O.exp(rands))
    dist = Mixture(dists, weights)

    # Call should return backend obj
    assert is_backend_distribution(dist())

    # Test sampling
    samples = dist.sample()
    assert is_backend_tensor(samples)
    assert samples.ndim == 1
    assert samples.shape[0] == 5
    samples = dist.sample(10)
    assert is_backend_tensor(samples)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 5

    # Test methods
    dist = Mixture(
        Normal(to_default_dtype([-1.0, 1.0]), to_default_dtype([1e-3, 1e-3])),
        to_default_dtype([0.5, 0.5]),
    )
    probs = dist.prob([-1.0, 1.0])
    assert np.isclose(probs[0] / probs[1], 1.0)

    dist = Mixture(
        Normal(to_default_dtype([-1.0, 1.0]), to_default_dtype([1e-3, 1e-3])),
        to_default_dtype(np.log(np.array([0.8, 0.2]))),
    )
    probs = dist.prob([-1.0, 1.0])
    assert np.isclose(probs[0] / probs[1], 4.0)

    dist = Mixture(
        Normal(to_default_dtype([-1.0, 1.0]), to_default_dtype([1e-3, 1e-3])),
        to_default_dtype(np.log(np.array([0.1, 0.9]))),
    )
    probs = dist.prob([-1.0, 1.0])
    assert np.isclose(probs[0] / probs[1], 1.0 / 9.0)

    # try w/ weight_type
    dist = Mixture(
        Normal(to_default_dtype([-1.0, 1.0]), to_default_dtype([1e-3, 1e-3])),
        logits=to_default_dtype(np.log(np.array([0.1, 0.9]))),
    )
    probs = dist.prob([-1.0, 1.0])
    assert np.isclose(probs[0] / probs[1], 1.0 / 9.0)

    dist = Mixture(
        Normal(to_default_dtype([-1.0, 1.0]), to_default_dtype([1e-3, 1e-3])),
        probs=to_default_dtype(np.array([0.1, 0.9])),
    )
    probs = dist.prob([-1.0, 1.0])
    assert np.isclose(probs[0] / probs[1], 1.0 / 9.0)
