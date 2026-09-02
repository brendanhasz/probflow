"""Typing hint utils."""

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import tensorflow as tf
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

else:
    TensorLike = Any
