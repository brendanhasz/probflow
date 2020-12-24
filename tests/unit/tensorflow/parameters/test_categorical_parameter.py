import numpy as np
import pytest
import tensorflow_probability as tfp

from probflow.parameters import CategoricalParameter

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_CategoricalParameter():
    """Tests probflow.parameters.CategoricalParameter"""

    # Should error with incorrect params
    with pytest.raises(TypeError):
        param = CategoricalParameter(k="a")
    with pytest.raises(ValueError):
        param = CategoricalParameter(k=1)

    # Create the parameter
    param = CategoricalParameter(k=3)

    # All samples should be 0, 1, or 2
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 1
    assert samples.shape[0] == 100
    assert all(s in [0, 1, 2] for s in samples.tolist())

    # 1D parameter
    param = CategoricalParameter(k=3, shape=5)
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert all(s in [0, 1, 2] for s in samples.flatten().tolist())

    # 2D parameter
    param = CategoricalParameter(k=3, shape=[5, 4])
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 3
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert samples.shape[2] == 4
    assert all(s in [0, 1, 2] for s in samples.flatten().tolist())
