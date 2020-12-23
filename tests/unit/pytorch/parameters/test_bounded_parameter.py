import numpy as np
import pytest

from probflow.parameters import BoundedParameter


def test_BoundedParameter():
    """Tests probflow.parameters.BoundedParameter"""

    # Should error with incorrect params
    with pytest.raises(ValueError):
        param = BoundedParameter(min=1.0, max=0.0)

    # Create the parameter
    param = BoundedParameter()

    # All samples should be between 0 and 1
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 1
    assert all(s > 0 and s < 1 for s in samples.flatten().tolist())

    # 1D parameter
    param = BoundedParameter(shape=5)
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert all(s > 0 and s < 1 for s in samples.flatten().tolist())

    # 2D parameter
    param = BoundedParameter(shape=[5, 4])
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 3
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert samples.shape[2] == 4
    assert all(s > 0 and s < 1 for s in samples.flatten().tolist())
