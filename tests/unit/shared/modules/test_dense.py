import numpy as np
import pytest

import probflow.utils.ops as O
from probflow.modules import Dense
from probflow.parameters import Parameter
from probflow.utils.casting import to_numpy
from probflow.utils.settings import Sampling
from probflow.utils.validation import is_backend_tensor


def test_Dense():
    """Tests probflow.modules.Dense."""
    # Should error w/ int < 1
    with pytest.raises(ValueError):
        dense = Dense(0, 1)
    with pytest.raises(ValueError):
        dense = Dense(5, -1)

    # Create the module
    dense = Dense(5, 1)

    # Test MAP outputs are same
    x = O.randn([4, 5])
    samples1 = dense(x)
    samples2 = dense(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # Test samples are different
    with Sampling(n=1):
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(to_numpy(samples1) != to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # parameters should return [weights, bias]
    param_list = dense.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in dense.parameters]
    assert "Dense_weights" in param_names
    assert "Dense_bias" in param_names
    weights = [p for p in dense.parameters if p.name == "Dense_weights"]
    assert weights[0].shape == [5, 1]
    bias = [p for p in dense.parameters if p.name == "Dense_bias"]
    assert bias[0].shape == [1, 1]

    # kl_loss should return sum of KL losses
    kl_loss = dense.kl_loss()
    assert is_backend_tensor(kl_loss)
    assert kl_loss.ndim == 0

    # test Flipout
    with Sampling(n=1, flipout=True):
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(to_numpy(samples1) != to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # With the probabilistic kwarg
    dense = Dense(5, 3, probabilistic=False)
    with Sampling(n=1):
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 3

    # With the weight and bias kwargs
    weight_kwargs = {"transform": O.exp}
    bias_kwargs = {"transform": O.softplus}
    dense = Dense(5, 2, weight_kwargs=weight_kwargs, bias_kwargs=bias_kwargs)
    with Sampling(n=1):
        samples1 = dense(x)
        samples2 = dense(x)
    assert np.all(to_numpy(samples1) != to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 2
