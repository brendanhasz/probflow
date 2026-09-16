"""Typing hint utils."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import jax
    import tensorflow as tf
    import tensorflow_probability as tfp
    import torch
    from tensorflow_probability.substrates import jax as tfp_jax

    from probflow.utils.jax_variable import JaxVariable

    TensorLike: TypeAlias = (
        int
        | float
        | list[int]
        | list[float]
        | np.ndarray
        | pd.DataFrame
        | pd.Series
        | tf.Tensor
        | tf.Variable
        | torch.Tensor
        | jax.Array
        | JaxVariable
    )
    ScalarLike: TypeAlias = (
        int
        | float
        | np.ndarray
        | tf.Tensor
        | tf.Variable
        | torch.Tensor
        | jax.Array
    )
    BackendTensor: TypeAlias = (
        tf.Tensor | tf.Variable | torch.Tensor | jax.Array | JaxVariable
    )
    BackendVariable: TypeAlias = tf.Variable | torch.nn.Parameter | JaxVariable
    BackendDataType: TypeAlias = tf.dtype | torch.dtype | np.dtype
    BackendDistribution: TypeAlias = (
        torch.distributions.distribution.Distribution
        | tfp.distributions.Distribution
        | tfp_jax.distributions.Distribution
    )

else:
    TensorLike = Any
    ScalarLike = Any
    BackendTensor = Any
    BackendVariable = Any
    BackendDataType = Any
    BackendDistribution = Any

TensorLike.__doc__ = (
    """A Tensor-like object (e.g., TensorFlow Tensor, numpy array, etc)."""
)
