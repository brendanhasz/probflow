import jax.numpy as jnp

from probflow.parameters import Parameter


def test_Parameter_slicing_jax():
    """Tests slicing a Parameter."""
    param = Parameter(shape=[2, 3, 4, 5])

    sliced = param[jnp.array([0]), :, ::2, :]
    assert sliced.ndim == 4
    assert sliced.shape == (1, 3, 2, 5)
