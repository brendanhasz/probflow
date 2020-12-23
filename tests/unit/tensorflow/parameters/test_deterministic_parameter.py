import numpy as np
import tensorflow_probability as tfp

from probflow.parameters import DeterministicParameter

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_DeterministicParameter():
    """Tests probflow.parameters.DeterministicParameter"""

    # Create the parameter
    param = DeterministicParameter()

    # All samples should be the same
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 1
    assert all(s == samples[0] for s in samples.flatten().tolist())

    # 1D parameter
    param = DeterministicParameter(shape=5)
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    for i in range(5):
        assert np.all(samples[:, i] == samples[0, i])

    # 2D parameter
    param = DeterministicParameter(shape=[5, 4])
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 3
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert samples.shape[2] == 4
    for i in range(5):
        for j in range(4):
            assert np.all(samples[:, i, j] == samples[0, i, j])
