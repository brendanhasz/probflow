import numpy as np

import probflow.utils.ops as O
from probflow.modules import BatchNormalization, Dense, Sequential
from probflow.parameters import Parameter
from probflow.utils.casting import to_numpy
from probflow.utils.settings import Sampling
from probflow.utils.validation import is_backend_tensor


def test_BatchNormalization():
    """Tests probflow.modules.BatchNormalization."""
    # Create the module
    bn = BatchNormalization(5)

    # Test MAP outputs are the same
    x = O.randn([4, 5])
    samples1 = bn(x)
    samples2 = bn(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 5

    # Samples should actually be the same b/c using deterministic posterior
    with Sampling():
        samples1 = bn(x)
        samples2 = bn(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 4
    assert samples1.shape[1] == 5

    # parameters should return list of all parameters
    param_list = bn.parameters
    assert isinstance(param_list, list)
    assert len(param_list) == 2
    assert all(isinstance(p, Parameter) for p in param_list)
    param_names = [p.name for p in bn.parameters]
    assert "BatchNormalization_weight" in param_names
    assert "BatchNormalization_bias" in param_names
    param_shapes = [p.shape for p in bn.parameters]
    assert [1, 5] in param_shapes

    # kl_loss should return sum of KL losses
    kl_loss = bn.kl_loss()
    assert is_backend_tensor(kl_loss)
    assert kl_loss.ndim == 0

    # Test it works w/ dense layer and sequential
    seq = Sequential(
        [
            Dense(5, 10),
            BatchNormalization(10),
            O.relu,
            Dense(10, 3),
            BatchNormalization(3),
            O.relu,
            Dense(3, 1),
        ]
    )
    assert len(seq.parameters) == 10
    out = seq(O.randn([6, 5]))
    assert out.ndim == 2
    assert out.shape[0] == 6
    assert out.shape[1] == 1


def test_BatchNormalization_2d():
    """Tests BatchNormalization with 2d inputs."""
    # Create the module
    bn = BatchNormalization([4, 3])

    # Test MAP outputs are the same
    x = O.randn([5, 4, 3])
    samples1 = bn(x)
    samples2 = bn(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 3
    assert samples1.shape[0] == 5
    assert samples1.shape[1] == 4
    assert samples1.shape[2] == 3
