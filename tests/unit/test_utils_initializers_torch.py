"""Tests probflow.utils.initializers w/ torch backend"""



import numpy as np
import torch

import probflow as pf
from probflow.utils import initializers



def test_xavier_torch():
    """Tests probflow.utils.initializers.xavier w/ torch backend"""

    pf.set_backend('pytorch')

    # Small array
    val1 = initializers.xavier([4, 3])
    assert isinstance(val1, torch.Tensor)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.xavier([400, 300])
    assert isinstance(val2, torch.Tensor)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.std(val1.numpy()) > np.std(val2.numpy())



def test_scale_xavier_torch():
    """Tests probflow.utils.initializers.scale_xavier w/ torch backend"""

    pf.set_backend('pytorch')

    # Small array
    val1 = initializers.scale_xavier([4, 3])
    assert isinstance(val1, torch.Tensor)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.scale_xavier([400, 300])
    assert isinstance(val2, torch.Tensor)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.mean(val1.numpy()) > np.mean(val2.numpy())



def test_pos_xavier_torch():
    """Tests probflow.utils.initializers.pos_xavier w/ torch backend"""

    pf.set_backend('pytorch')

    # Small array
    val1 = initializers.pos_xavier([4, 3])
    assert isinstance(val1, torch.Tensor)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.pos_xavier([400, 300])
    assert isinstance(val2, torch.Tensor)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.mean(val1.numpy()) < np.mean(val2.numpy())

