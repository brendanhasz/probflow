"""Tests probflow.utils.initializers."""

import numpy as np

from probflow.utils import initializers
from probflow.utils.casting import to_numpy
from probflow.utils.validation import is_backend_tensor


def test_xavier():
    """Tests probflow.utils.initializers.xavier."""
    # Small array
    val1 = initializers.xavier([4, 3])
    assert is_backend_tensor(val1)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.xavier([400, 300])
    assert is_backend_tensor(val2)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.std(to_numpy(val1)) > np.std(to_numpy(val2))


def test_scale_xavier():
    """Tests probflow.utils.initializers.scale_xavier."""
    # Small array
    val1 = initializers.scale_xavier([4, 3])
    assert is_backend_tensor(val1)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.scale_xavier([400, 300])
    assert is_backend_tensor(val2)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.mean(to_numpy(val1)) > np.mean(to_numpy(val2))


def test_pos_xavier():
    """Tests probflow.utils.initializers.pos_xavier."""
    # Small array
    val1 = initializers.pos_xavier([4, 3])
    assert is_backend_tensor(val1)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.pos_xavier([400, 300])
    assert is_backend_tensor(val2)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.mean(to_numpy(val1)) < np.mean(to_numpy(val2))
