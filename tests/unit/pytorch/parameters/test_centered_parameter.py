import numpy as np
import pytest
import torch

from probflow.distributions import Bernoulli
from probflow.models import Model
from probflow.parameters import CenteredParameter


def is_close(a, b, tol=1e-5):
    return np.abs(a - b) < tol


def test_CenteredParameter_all_1d():
    """Tests probflow.parameters.CenteredParameter w/ center_by=all + 1D """

    # Create the parameter
    param = CenteredParameter(5)

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample2.shape[0] == 5
    assert sample1.shape[1] == 1
    assert sample2.shape[1] == 1
    assert np.all(sample1 == sample2)

    # mean should be 0! (that's the point of a centered parameter!)
    assert is_close(np.mean(sample1), 0)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 1
    assert sample2.shape[0] == 5
    assert sample2.shape[1] == 1
    assert np.all(sample1 != sample2)

    # mean should be 0! (that's the point of a centered parameter!)
    assert is_close(np.mean(sample1), 0)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 3
    assert sample2.ndim == 3
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 5
    assert sample1.shape[2] == 1
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 5
    assert sample2.shape[2] == 1
    assert np.all(sample1 != sample2)

    # mean should be 0!
    assert np.all(np.abs(np.mean(sample1, axis=1)) < 1e-5)


def test_CenteredParameter_all_2d():
    """Tests probflow.parameters.CenteredParameter w/ center_by=all + 2D """

    # Shouldn't allow >2 dims
    with pytest.raises(ValueError):
        param = CenteredParameter([5, 6, 7])

    # Create the parameter
    param = CenteredParameter([5, 6])

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample2.shape[0] == 5
    assert sample1.shape[1] == 6
    assert sample2.shape[1] == 6
    assert np.all(sample1 == sample2)

    # mean should be 0! (that's the point of a centered parameter!)
    assert is_close(np.mean(sample1), 0)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 6
    assert sample2.shape[0] == 5
    assert sample2.shape[1] == 6
    assert np.all(sample1 != sample2)

    # mean should be 0!
    assert is_close(np.mean(sample1), 0)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 3
    assert sample2.ndim == 3
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 5
    assert sample1.shape[2] == 6
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 5
    assert sample2.shape[2] == 6
    assert np.all(sample1 != sample2)

    # mean should be 0!
    assert np.all(np.abs(np.mean(sample1.reshape((10, -1)), axis=1)) < 1e-5)


def test_CenteredParameter_column():
    """Tests probflow.parameters.CenteredParameter w/ center_by=column + 2D """

    # Create the parameter
    param = CenteredParameter([5, 6], center_by="column")

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample2.shape[0] == 5
    assert sample1.shape[1] == 6
    assert sample2.shape[1] == 6
    assert np.all(sample1 == sample2)

    # mean of each column should be 0
    assert np.all(np.abs(np.mean(sample1, axis=0)) < 1e-5)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 6
    assert sample2.shape[0] == 5
    assert sample2.shape[1] == 6
    assert np.all(sample1 != sample2)

    # mean of each column should be 0
    assert np.all(np.abs(np.mean(sample1, axis=0)) < 1e-5)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 3
    assert sample2.ndim == 3
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 5
    assert sample1.shape[2] == 6
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 5
    assert sample2.shape[2] == 6
    assert np.all(sample1 != sample2)

    # mean of each column for each sample should be 0
    assert np.all(np.abs(np.mean(sample1, axis=1)) < 1e-5)


def test_CenteredParameter_row():
    """Tests probflow.parameters.CenteredParameter w/ center_by=row + 2D """

    # Create the parameter
    param = CenteredParameter([5, 6], center_by="row")

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample2.shape[0] == 5
    assert sample1.shape[1] == 6
    assert sample2.shape[1] == 6
    assert np.all(sample1 == sample2)

    # mean of each column should be 0
    assert np.all(np.abs(np.mean(sample1, axis=1)) < 1e-5)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 6
    assert sample2.shape[0] == 5
    assert sample2.shape[1] == 6
    assert np.all(sample1 != sample2)

    # mean of each column should be 0
    assert np.all(np.abs(np.mean(sample1, axis=1)) < 1e-5)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 3
    assert sample2.ndim == 3
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 5
    assert sample1.shape[2] == 6
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 5
    assert sample2.shape[2] == 6
    assert np.all(sample1 != sample2)

    # mean of each column for each sample should be 0
    assert np.all(np.abs(np.mean(sample1, axis=2)) < 1e-5)


def test_CenteredParameter_fit():
    """Tests fitting a model with probflow.parameters.CenteredParameter"""

    class MyModel(Model):
        def __init__(self, di, do):
            self.w = CenteredParameter([di, do], center_by="column")

        def __call__(self, x):
            return Bernoulli(torch.tensor(x) @ self.w())

    N = 128
    Di = 5
    Do = 3
    x = np.random.randn(N, Di).astype("float32")
    w = np.random.randn(Di, Do).astype("float32")
    y = x @ w + 0.1 * np.random.randn(N, Do)
    y = (1/(1+np.exp(-y)) > np.random.rand(N, Do)).astype('float32')

    model = MyModel(Di, Do)

    model.fit(x, y, epochs=2, batch_size=128, eager=True)
