import numpy as np

import probflow.utils.ops as O
from probflow.modules import Dense, Sequential
from probflow.parameters import Parameter
from probflow.utils.casting import to_numpy
from probflow.utils.settings import Sampling
from probflow.utils.validation import is_backend_tensor


def test_Sequential():
    """Tests probflow.modules.Sequential."""
    # Create the module
    seq = Sequential([Dense(5, 10), O.relu, Dense(10, 3), O.relu, Dense(3, 1)])

    # Steps should be list
    assert isinstance(seq.steps, list)
    assert len(seq.steps) == 5

    # Test MAP outputs are the same
    x = O.randn([4, 5])
    samples1 = seq(x)
    samples2 = seq(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # Test samples are different
    with Sampling(n=1):
        samples1 = seq(x)
        samples2 = seq(x)
    assert np.all(to_numpy(samples1) != to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 1

    # parameters should return list of all parameters
    param_list = seq.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 6
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in seq.parameters]
    assert "Dense_weights" in param_names
    assert "Dense_bias" in param_names
    param_shapes = [p.shape for p in seq.parameters]
    assert [5, 10] in param_shapes
    assert [1, 10] in param_shapes
    assert [10, 3] in param_shapes
    assert [1, 3] in param_shapes
    assert [3, 1] in param_shapes
    assert [1, 1] in param_shapes

    # kl_loss should return sum of KL losses
    kl_loss = seq.kl_loss()
    assert is_backend_tensor(kl_loss)
    assert kl_loss.ndim == 0
