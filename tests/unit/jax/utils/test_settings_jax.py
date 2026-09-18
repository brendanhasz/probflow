import jax.numpy as jnp
import numpy as np
import pytest

import probflow as pf


def test_datatype():
    """Tests get and set_datatype for JAX."""
    assert np.dtype(pf.get_datatype()) == np.dtype(jnp.float32)

    pf.set_datatype(jnp.float64)
    assert np.dtype(pf.get_datatype()) == np.dtype(jnp.float64)
    pf.set_datatype(jnp.float32)

    with pytest.raises(TypeError):
        pf.set_datatype("lala")


def test_jax_seed():
    """Tests reproducible JAX randomness through the public seed helper."""
    pf.set_jax_seed(123)
    first = np.asarray(pf.utils.ops.randn([3]))
    pf.set_jax_seed(123)
    second = np.asarray(pf.utils.ops.randn([3]))
    assert np.array_equal(first, second)
