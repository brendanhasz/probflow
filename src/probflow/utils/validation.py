"""
The utils.validation module contains functions for checking that inputs have
the correct type.

* :func:`.ensure_tensor_like`

----------

"""


import numpy as np

from probflow.utils.base import BaseParameter
from probflow.utils.settings import get_backend


def ensure_tensor_like(obj, name):
    """Determine whether an object can be cast to a Tensor"""

    # Check for non-backend-dependent types
    if isinstance(obj, (int, float, np.ndarray, list)):
        return

    # Check for backend-dependent types
    if get_backend() == "pytorch":
        import torch

        tensor_types = (torch.Tensor, BaseParameter)
    else:
        import tensorflow as tf

        tensor_types = (tf.Tensor, tf.Variable, BaseParameter)
    if not isinstance(obj, tensor_types):
        raise TypeError(name + " must be Tensor-like")
