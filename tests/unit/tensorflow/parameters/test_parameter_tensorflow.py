"""Test the generic Parameter when backend = tensorflow."""
import tensorflow as tf

from probflow.parameters import Parameter
from probflow.utils.casting import to_numpy



def test_Parameter_slicing():
    """Tests a slicing Parameters."""
    # Create 1D parameter
    param = Parameter(shape=[2, 3, 4, 5])

    sl = to_numpy(param[tf.constant([0]), :, ::2, :])
    assert sl.ndim == 4
    assert sl.shape[0] == 1
    assert sl.shape[1] == 3
    assert sl.shape[2] == 2
    assert sl.shape[3] == 5