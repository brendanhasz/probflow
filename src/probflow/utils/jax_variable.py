"""A mutable variable wrapper for the JAX backend.

JAX arrays are immutable and carry no attached autograd graph, unlike
``tf.Variable`` or ``torch.nn.Parameter``. :class:`.JaxVariable` stands in
for those two, holding its current value in a mutable ``.value`` attribute
that gets overwritten in place by the optimizer/training step. It implements
the ``__jax_array__`` protocol so it is automatically coerced to a plain
array anywhere JAX or TensorFlow Probability's JAX substrate expect one.
"""

from typing import Any

__all__ = ["JaxVariable"]


class JaxVariable:
    """Mutable container standing in for a trainable JAX array."""

    def __init__(self, initial_value: Any) -> None:
        import jax.numpy as jnp

        self.value = jnp.asarray(initial_value)

    def __jax_array__(self) -> Any:
        return self.value

    def __array__(self, dtype: Any = None) -> Any:
        import numpy as np

        return np.asarray(self.value, dtype=dtype)

    @property
    def shape(self) -> Any:
        """Get the shape of the variable."""
        return self.value.shape

    @property
    def dtype(self) -> Any:
        """Get the data type of the variable."""
        return self.value.dtype

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the variable."""
        return int(self.value.ndim)

    def __repr__(self) -> str:
        return f"JaxVariable({self.value!r})"

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __neg__(self) -> Any:
        return -self.value

    def __add__(self, other: Any) -> Any:
        return self.value + other

    def __radd__(self, other: Any) -> Any:
        return other + self.value

    def __sub__(self, other: Any) -> Any:
        return self.value - other

    def __rsub__(self, other: Any) -> Any:
        return other - self.value

    def __mul__(self, other: Any) -> Any:
        return self.value * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.value

    def __truediv__(self, other: Any) -> Any:
        return self.value / other

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.value

    def __pow__(self, other: Any) -> Any:
        return self.value**other

    def __matmul__(self, other: Any) -> Any:
        return self.value @ other

    def __rmatmul__(self, other: Any) -> Any:
        return other @ self.value
