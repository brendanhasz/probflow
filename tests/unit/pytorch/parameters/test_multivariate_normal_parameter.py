import numpy as np
import torch

from probflow.parameters import MultivariateNormalParameter


def test_MultivariateNormalParameter():
    """Tests probflow.parameters.MultivariateNormalParameter"""

    # Create the parameter
    param = MultivariateNormalParameter(4)

    # kl_loss should still be scalar
    kl_loss = param.kl_loss()
    assert isinstance(kl_loss, torch.Tensor)
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
    assert isinstance(s, torch.Tensor)
    assert s.ndim == 2
    assert s.shape[0] == 2
    assert s.shape[1] == 1
    s = param[1]
    assert isinstance(s, torch.Tensor)
    assert s.ndim == 2
    assert s.shape[0] == 1
    assert s.shape[1] == 1
    s = param[-1]
    assert isinstance(s, torch.Tensor)
    assert s.ndim == 2
    assert s.shape[0] == 1
    assert s.shape[1] == 1
