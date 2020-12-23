import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.parameters import ScaleParameter
from probflow.utils.base import BaseDistribution
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_ScaleParameter():
    """Tests probflow.parameters.ScaleParameter"""

    # Create the parameter
    param = ScaleParameter()

    # All samples should be > 0
    assert np.all(param.posterior_sample(n=1000) > 0)

    # 1D ScaleParameter
    param = ScaleParameter(shape=5)
    samples = param.posterior_sample(n=10)
    assert samples.ndim == 2
    assert samples.shape[0] == 10
    assert samples.shape[1] == 5
    assert np.all(samples > 0)

    # 2D ScaleParameter
    param = ScaleParameter(shape=[5, 4])
    samples = param.posterior_sample(n=10)
    assert samples.ndim == 3
    assert samples.shape[0] == 10
    assert samples.shape[1] == 5
    assert samples.shape[2] == 4
    assert np.all(samples > 0)
