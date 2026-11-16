import numpy as np

from probflow.distributions import Wishart
from probflow.parameters import WishartParameter
from probflow.utils.casting import to_default_dtype, to_numpy
from probflow.utils.validation import is_backend_tensor


def _is_symmetric_pd(samples):
    samples = np.asarray(samples)
    assert np.allclose(samples, np.swapaxes(samples, -1, -2), atol=1e-4)
    eigvals = np.linalg.eigvalsh((samples + np.swapaxes(samples, -1, -2)) / 2)
    return bool(np.all(eigvals > 0))


def test_WishartParameter_samples():
    """Tests probflow.parameters.WishartParameter sample shapes."""
    d = 4
    param = WishartParameter(d)

    # posterior_mean is deterministic
    mean1 = param.posterior_mean()
    mean2 = param.posterior_mean()
    assert mean1.shape[-2:] == (d, d)
    assert np.all(mean1 == mean2)

    # posterior_sample returns random d x d matrices
    sample1 = param.posterior_sample()
    sample2 = param.posterior_sample()
    assert sample1.shape[-2:] == (d, d)
    assert np.all(sample1 != sample2)

    # Multiple samples
    samples = param.posterior_sample(10)
    assert samples.ndim == 4
    assert samples.shape[0] == 10
    assert samples.shape[-2:] == (d, d)

    # Prior samples
    prior_sample = param.prior_sample()
    assert prior_sample.ndim == 2
    assert prior_sample.shape == (d, d)
    prior_samples = param.prior_sample(n=7)
    assert prior_samples.ndim == 3
    assert prior_samples.shape == (7, d, d)


def test_WishartParameter_default_prior():
    """Default prior is a Wishart, custom priors are respected."""
    param = WishartParameter(3)
    assert isinstance(param.prior, Wishart)

    custom = Wishart(10, to_default_dtype(np.eye(3) * 2))
    param = WishartParameter(3, prior=custom)
    assert param.prior is custom
    assert param.prior_sample().shape == (3, 3)


def test_WishartParameter_kl_loss():
    param = WishartParameter(3)
    loss = param.kl_loss()
    assert is_backend_tensor(loss)
    assert loss.ndim == 0
    assert np.isfinite(to_numpy(loss))
    assert to_numpy(loss) >= 0

    param.bayesian_update()
    np.testing.assert_allclose(to_numpy(param.kl_loss()), 0, atol=1e-5)


def test_WishartParameter_samples_are_covariance_matrices():
    """Samples should be symmetric PD covariance (not correlation) matrices."""
    d = 3
    param = WishartParameter(d)
    samples = np.asarray(param.posterior_sample(100))

    # Symmetric positive definite
    assert _is_symmetric_pd(samples)

    # Diagonals are variances, so they should not be (all) exactly 1.0
    diags = np.diagonal(samples, axis1=-2, axis2=-1)
    assert diags.shape[-1] == d
    assert np.all(diags > 0)
    assert not np.allclose(diags, 1.0)
    assert not np.any(diags == 1.0)

    # Also true for the posterior mean
    mean = np.asarray(param.posterior_mean())
    assert _is_symmetric_pd(mean)
    assert not np.any(np.diagonal(mean, axis1=-2, axis2=-1) == 1.0)

    # Large-scale prior should give clearly non-unit diagonals
    big = Wishart(d + 1, to_default_dtype(np.eye(d) * 100.0))
    param_big = WishartParameter(d, prior=big)
    prior_samples = np.asarray(param_big.prior_sample(50))
    assert _is_symmetric_pd(prior_samples)
    assert np.mean(np.diagonal(prior_samples, axis1=-2, axis2=-1)) > 5.0


def test_WishartParameter_output_tril():
    """With output_tril=True, samples are lower triangular Cholesky factors."""
    d = 3
    param = WishartParameter(d, output_tril=True)
    samples = np.asarray(param.posterior_sample(20))
    assert samples.shape[-2:] == (d, d)

    # Upper triangle is zero, diagonal positive
    assert np.allclose(samples, np.tril(samples))
    assert np.all(np.diagonal(samples, axis1=-2, axis2=-1) > 0)

    # Reconstructed L L^T is a valid covariance matrix
    cov = samples @ np.swapaxes(samples, -1, -2)
    assert _is_symmetric_pd(cov)
