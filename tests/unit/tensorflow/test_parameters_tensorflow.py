"""Tests the probflow.parameters module when backend = tensorflow"""


import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from probflow.core.settings import Sampling
from probflow.parameters import *
from probflow.distributions import BaseDistribution

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Parameter_scalar():
    """Tests the generic scalar Parameter"""

    # Create scalar parameter
    param = Parameter()

    # Check defaults
    assert isinstance(param.shape, list)
    assert param.shape[0] == 1
    assert isinstance(param.untransformed_variables, dict)
    assert all(isinstance(p, str) for p in param.untransformed_variables)
    assert all(
        isinstance(p, tf.Variable)
        for _, p in param.untransformed_variables.items()
    )

    # Shape should be >0
    with pytest.raises(ValueError):
        Parameter(shape=-1)
    with pytest.raises(ValueError):
        Parameter(shape=[20, 0, 1])

    # trainable_variables should be a property returning list of vars
    assert all(isinstance(v, tf.Variable) for v in param.trainable_variables)

    # variables should be a property returning dict of transformed vars
    assert isinstance(param.variables, dict)
    assert all(isinstance(v, str) for v in param.variables)

    # loc should be variable, while scale should have been transformed->tensor
    assert isinstance(param.variables["loc"], tf.Variable)
    assert isinstance(param.variables["scale"], tf.Tensor)

    # posterior should be a distribution object
    assert isinstance(param.posterior, BaseDistribution)
    assert isinstance(param.posterior(), tfd.Normal)

    # __call__ should return the MAP estimate by default
    sample1 = param()
    sample2 = param()
    assert sample1.ndim == 1
    assert sample2.ndim == 1
    assert sample1.shape[0] == 1
    assert sample2.shape[0] == 1
    assert sample1.numpy() == sample2.numpy()

    # within a Sampling statement, should randomly sample from the dist
    with Sampling():
        sample1 = param()
        sample2 = param()
    assert sample1.ndim == 1
    assert sample2.ndim == 1
    assert sample1.shape[0] == 1
    assert sample2.shape[0] == 1
    assert sample1.numpy() != sample2.numpy()

    # sampling statement should effect N samples
    with Sampling(n=10):
        sample1 = param()
        sample2 = param()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 1
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 1
    assert np.all(sample1.numpy() != sample2.numpy())

    # kl_loss should return sum of kl divergences
    kl_loss = param.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # prior_sample should be 1D
    prior_sample = param.prior_sample()
    assert prior_sample.ndim == 0
    prior_sample = param.prior_sample(n=7)
    assert prior_sample.ndim == 1
    assert prior_sample.shape[0] == 7


def test_Parameter_no_prior():
    """Tests a parameter with no prior"""

    # Create parameter with no prior
    param = Parameter(prior=None)

    # kl_loss should return 0
    kl_loss = param.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0
    assert kl_loss.numpy() == 0.0


def test_Parameter_1D():
    """Tests a 1D Parameter"""

    # Create 1D parameter
    param = Parameter(shape=5)

    # kl_loss should still be scalar
    kl_loss = param.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 1
    assert sample2.ndim == 1
    assert sample1.shape[0] == 5
    assert sample2.shape[0] == 5
    assert np.all(sample1 == sample2)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 1
    assert sample2.ndim == 1
    assert sample1.shape[0] == 5
    assert sample2.shape[0] == 5
    assert np.all(sample1 != sample2)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 5
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 5
    assert np.all(sample1 != sample2)

    # prior_sample should still be 1D
    prior_sample = param.prior_sample()
    assert prior_sample.ndim == 0
    prior_sample = param.prior_sample(n=7)
    assert prior_sample.ndim == 1
    assert prior_sample.shape[0] == 7

    # n_parameters property
    nparams = param.n_parameters
    assert isinstance(nparams, int)
    assert nparams == 5


def test_Parameter_2D():
    """Tests a 2D Parameter"""

    # Create 1D parameter
    param = Parameter(shape=[5, 4], name="lala")

    # repr
    pstr = param.__repr__()
    assert pstr == "<pf.Parameter lala shape=[5, 4]>"

    # kl_loss should still be scalar
    kl_loss = param.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 4
    assert sample2.shape[0] == 5
    assert sample2.shape[1] == 4
    assert np.all(sample1 == sample2)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 5
    assert sample1.shape[1] == 4
    assert sample2.shape[0] == 5
    assert sample2.shape[1] == 4
    assert np.all(sample1 != sample2)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 3
    assert sample2.ndim == 3
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 5
    assert sample1.shape[2] == 4
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 5
    assert sample2.shape[2] == 4
    assert np.all(sample1 != sample2)

    # prior_sample should still be 1D
    prior_sample = param.prior_sample()
    assert prior_sample.ndim == 0
    prior_sample = param.prior_sample(n=7)
    assert prior_sample.ndim == 1
    assert prior_sample.shape[0] == 7

    # n_parameters property
    nparams = param.n_parameters
    assert isinstance(nparams, int)
    assert nparams == 20


def test_Parameter_slicing():
    """Tests a slicing Parameters"""

    # Create 1D parameter
    param = Parameter(shape=[2, 3, 4, 5])

    # Should be able to slice!
    sl = param[0].numpy()
    assert sl.ndim == 4
    assert sl.shape[0] == 1
    assert sl.shape[1] == 3
    assert sl.shape[2] == 4
    assert sl.shape[3] == 5

    sl = param[:, :, :, :2].numpy()
    assert sl.ndim == 4
    assert sl.shape[0] == 2
    assert sl.shape[1] == 3
    assert sl.shape[2] == 4
    assert sl.shape[3] == 2

    sl = param[1, ..., :2].numpy()
    assert sl.ndim == 4
    assert sl.shape[0] == 1
    assert sl.shape[1] == 3
    assert sl.shape[2] == 4
    assert sl.shape[3] == 2

    sl = param[...].numpy()
    assert sl.ndim == 4
    assert sl.shape[0] == 2
    assert sl.shape[1] == 3
    assert sl.shape[2] == 4
    assert sl.shape[3] == 5

    sl = param[tf.constant([0]), :, ::2, :].numpy()
    assert sl.ndim == 4
    assert sl.shape[0] == 1
    assert sl.shape[1] == 3
    assert sl.shape[2] == 2
    assert sl.shape[3] == 5


def test_Parameter_posterior_ci():
    """Tests probflow.parameters.Parameter.posterior_ci"""

    # With a scalar parameter
    param = Parameter()
    lb, ub = param.posterior_ci()
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 1
    assert ub.ndim == 1
    assert lb.shape[0] == 1
    assert ub.shape[0] == 1

    # Should error w/ invalid ci or n vals
    with pytest.raises(ValueError):
        lb, ub = param.posterior_ci(ci=-0.1)
    with pytest.raises(ValueError):
        lb, ub = param.posterior_ci(ci=1.1)
    with pytest.raises(ValueError):
        lb, ub = param.posterior_ci(n=0)

    # With a 1D parameter
    param = Parameter(shape=5)
    lb, ub = param.posterior_ci()
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 1
    assert ub.ndim == 1
    assert lb.shape[0] == 5
    assert ub.shape[0] == 5

    # With a 2D parameter
    param = Parameter(shape=[5, 4])
    lb, ub = param.posterior_ci()
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert ub.ndim == 2
    assert lb.shape[0] == 5
    assert lb.shape[1] == 4
    assert ub.shape[0] == 5
    assert ub.shape[1] == 4


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


def test_MultivariateNormalParameter():
    """Tests probflow.parameters.MultivariateNormalParameter"""

    # Create the parameter
    param = MultivariateNormalParameter(4)

    # kl_loss should still be scalar
    kl_loss = param.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # posterior_mean should return mean
    sample1 = param.posterior_mean()
    sample2 = param.posterior_mean()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 4
    assert sample2.shape[0] == 4
    assert sample1.shape[1] == 1
    assert sample2.shape[1] == 1
    assert np.all(sample1 == sample2)

    # posterior_sample should return samples
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.ndim == 2
    assert sample2.ndim == 2
    assert sample1.shape[0] == 4
    assert sample2.shape[0] == 4
    assert np.all(sample1 != sample2)

    # posterior_sample should be able to return multiple samples
    sample1 = param.posterior_sample(10)
    sample2 = param.posterior_sample(10)
    assert sample1.ndim == 3
    assert sample2.ndim == 3
    assert sample1.shape[0] == 10
    assert sample1.shape[1] == 4
    assert sample2.shape[0] == 10
    assert sample2.shape[1] == 4
    assert np.all(sample1 != sample2)

    # prior_sample should be d-dimensional
    prior_sample = param.prior_sample()
    assert prior_sample.ndim == 2
    assert prior_sample.shape[0] == 4
    prior_sample = param.prior_sample(n=7)
    assert prior_sample.ndim == 3
    assert prior_sample.shape[0] == 7
    assert prior_sample.shape[1] == 4

    # test slicing
    s = param[:-2]
    assert isinstance(s, tf.Tensor)
    assert s.ndim == 2
    assert s.shape[0] == 2
    assert s.shape[1] == 1
    s = param[1]
    assert isinstance(s, tf.Tensor)
    assert s.ndim == 2
    assert s.shape[0] == 1
    assert s.shape[1] == 1
    s = param[-1]
    assert isinstance(s, tf.Tensor)
    assert s.ndim == 2
    assert s.shape[0] == 1
    assert s.shape[1] == 1
