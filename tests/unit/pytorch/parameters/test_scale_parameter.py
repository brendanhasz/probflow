import numpy as np

from probflow.parameters import ScaleParameter


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
