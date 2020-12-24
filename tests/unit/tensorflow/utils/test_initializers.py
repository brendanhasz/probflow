"""Tests probflow.utils.initializers"""


import numpy as np
import tensorflow as tf

from probflow.utils import initializers


def test_xavier():
    """Tests probflow.utils.initializers.xavier"""

    # Small array
    val1 = initializers.xavier([4, 3])
    assert isinstance(val1, tf.Tensor)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.xavier([400, 300])
    assert isinstance(val2, tf.Tensor)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.std(val1.numpy()) > np.std(val2.numpy())


def test_scale_xavier():
    """Tests probflow.utils.initializers.scale_xavier"""

    # Small array
    val1 = initializers.scale_xavier([4, 3])
    assert isinstance(val1, tf.Tensor)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.scale_xavier([400, 300])
    assert isinstance(val2, tf.Tensor)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.mean(val1.numpy()) > np.mean(val2.numpy())


def test_pos_xavier():
    """Tests probflow.utils.initializers.pos_xavier"""

    # Small array
    val1 = initializers.pos_xavier([4, 3])
    assert isinstance(val1, tf.Tensor)
    assert val1.ndim == 2
    assert val1.shape[0] == 4
    assert val1.shape[1] == 3

    # Large array
    val2 = initializers.pos_xavier([400, 300])
    assert isinstance(val2, tf.Tensor)
    assert val2.ndim == 2
    assert val2.shape[0] == 400
    assert val2.shape[1] == 300

    # Large array should have smaller value spread
    assert np.mean(val1.numpy()) < np.mean(val2.numpy())
