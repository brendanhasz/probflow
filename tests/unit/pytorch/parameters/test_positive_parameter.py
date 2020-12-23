from probflow.parameters import PositiveParameter


def test_PositiveParameter():
    """Tests probflow.parameters.PositiveParameter"""

    # Create the parameter
    param = PositiveParameter()

    # All samples should be between 0 and 1
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 1
    assert all(s > 0 for s in samples.flatten().tolist())

    # 1D parameter
    param = PositiveParameter(shape=5)
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 2
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert all(s > 0 for s in samples.flatten().tolist())

    # 2D parameter
    param = PositiveParameter(shape=[5, 4])
    samples = param.posterior_sample(n=100)
    assert samples.ndim == 3
    assert samples.shape[0] == 100
    assert samples.shape[1] == 5
    assert samples.shape[2] == 4
    assert all(s > 0 for s in samples.flatten().tolist())
