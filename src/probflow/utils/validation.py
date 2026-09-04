"""
The utils.validation module contains functions for checking that inputs have
the correct type.

* :func:`.ensure_tensor_like`

----------

"""

from typing import Any

import numpy as np
import pandas as pd

from probflow.utils.base import BaseParameter
from probflow.utils.settings import get_backend


def ensure_tensor_like(obj: Any, name: str) -> None:
    """Determine whether an object can be cast to a Tensor"""

    # Check for non-backend-dependent types
    if isinstance(
        obj,
        (
            int,
            float,
            np.ndarray,
            pd.DataFrame,
            pd.Series,
        ),
    ):
        return
    if isinstance(obj, list):
        for o in obj:
            ensure_tensor_like(o, name)
        return

    # Check for backend-dependent types
    if get_backend() == "pytorch":
        import torch

        if not isinstance(obj, (torch.Tensor, BaseParameter)):
            raise TypeError(name + " must be Tensor-like")
    else:
        import tensorflow as tf

        if not isinstance(obj, (tf.Tensor, tf.Variable, BaseParameter)):
            raise TypeError(name + " must be Tensor-like")
