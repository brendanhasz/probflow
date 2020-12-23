import pytest

from probflow.parameters import DirichletParameter


def test_DirichletParameter():
    """Tests probflow.parameters.DirichletParameter"""

    # Should error with incorrect params
    with pytest.raises(TypeError):
        param = DirichletParameter(k="a")
    with pytest.raises(ValueError):
        param = DirichletParameter(k=1)

    # Create the parameter
    param = DirichletParameter(k=3)

    # All samples should be between 0 and 1
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 3
    assert all(s > 0 and s < 1 for s in samples.flatten().tolist())

    # 1D parameter
    param = DirichletParameter(k=3, shape=5)
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 3
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert samples.shape[2] == 3
    assert all(s > 0 and s < 1 for s in samples.flatten().tolist())

    # 2D parameter
    param = DirichletParameter(k=3, shape=[5, 4])
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 4
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert samples.shape[2] == 4
    assert samples.shape[3] == 3
    assert all(s > 0 and s < 1 for s in samples.flatten().tolist())
