"""Typing hint utils."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import tensorflow as tf
    import tensorflow_probability as tfp
    import torch

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
    )
    ScalarLike: TypeAlias = (
        int | float | np.ndarray | tf.Tensor | tf.Variable | torch.Tensor
    )
    BackendTensor: TypeAlias = tf.Tensor | tf.Variable | torch.Tensor
    BackendVariable: TypeAlias = tf.Variable | torch.nn.Parameter
    BackendDataType: TypeAlias = tf.dtype | torch.dtype
    BackendDistribution: TypeAlias = (
        torch.distributions.distribution.Distribution
        | tfp.distributions.Distribution
    )

else:
    TensorLike = Any
    ScalarLike = Any
    BackendTensor = Any
    BackendVariable = Any
    BackendDataType = Any
    BackendDistribution = Any
