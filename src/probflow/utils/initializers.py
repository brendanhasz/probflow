"""Initializers.

Functions to initialize posterior distribution variables.

* :func:`.xavier` - Xavier initializer
* :func:`.scale_xavier` - Xavier initializer scaled for scale parameters
* :func:`.pos_xavier` - positive-only initizlier

----------

"""

from collections.abc import Callable

import numpy as np

from probflow.utils.settings import ProbflowBackend, get_backend, get_datatype
from probflow.utils.typing import BackendTensor, ScalarLike


def xavier(shape: list[int] | tuple[int, ...]) -> BackendTensor:
    """Xavier initializer."""
    scale = np.sqrt(2 / sum(shape))
    if get_backend() == ProbflowBackend.PYTORCH:
        # TODO: use truncated normal for torch
        import torch

        x = torch.empty(shape)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=scale)
        return x
        # return torch.randn(shape, dtype=get_datatype()) * scale
    elif get_backend() == ProbflowBackend.JAX:
        import jax

        from probflow.utils.settings import _next_jax_key

        return (
            jax.random.truncated_normal(
                _next_jax_key(), -2.0, 2.0, shape, dtype=get_datatype()
            )
            * scale
        )
    else:
        import tensorflow as tf

        return tf.random.truncated_normal(
            shape, mean=0.0, stddev=scale, dtype=get_datatype()
        )


def scale_xavier(shape: list[int] | tuple[int, ...]) -> BackendTensor:
    """Xavier initializer for scale variables."""
    vals = xavier(shape)
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        numel = torch.prod(torch.Tensor(shape))
        return vals + 2 - 2 * torch.log(numel) / np.log(10.0)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        numel = float(jnp.prod(jnp.array(shape)))
        return vals + 2 - 2 * np.log(numel) / np.log(10.0)
    else:
        import tensorflow as tf

        numel = float(tf.reduce_prod(shape))
        return vals + 2 - 2 * tf.math.log(numel) / tf.math.log(10.0)


def pos_xavier(shape: list[int] | tuple[int, ...]) -> BackendTensor:
    """Xavier initializer for positive variables."""
    vals = xavier(shape)
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        numel = torch.prod(torch.Tensor(shape))
        return vals + torch.log(numel) / np.log(10.0)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        numel = float(jnp.prod(jnp.array(shape)))
        return vals + np.log(numel) / np.log(10.0)
    else:
        import tensorflow as tf

        numel = float(tf.reduce_prod(shape))
        return vals + tf.math.log(numel) / tf.math.log(10.0)


def full_of(
    val: ScalarLike,
) -> Callable[[list[int] | tuple[int, ...]], BackendTensor]:
    """Get initializer which returns tensor full of single value."""
    import probflow.utils.ops as O

    def init(shape: list[int] | tuple[int, ...]) -> BackendTensor:
        return val * O.ones(shape)

    return init
