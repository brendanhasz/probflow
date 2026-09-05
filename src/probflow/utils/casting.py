"""Functions for casting back and forth betweeen Tensors and numpy arrays.

* :func:`.to_numpy`
* :func:`.to_tensor`
* :func:`.to_default_dtype`
* :func:`.make_input_tensor`

----------

"""

__all__ = [
    "make_input_tensor",
    "to_default_dtype",
    "to_numpy",
    "to_tensor",
]

from collections.abc import Callable
from typing import Concatenate, ParamSpec

import numpy as np
import pandas as pd

from probflow.utils.settings import get_backend, get_datatype
from probflow.utils.typing import BackendTensor, TensorLike

P = ParamSpec("P")


def to_numpy(x: TensorLike) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, list):
        return [to_numpy(e) for e in x]
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    elif get_backend() == "tensorflow":
        import tensorflow as tf

        if isinstance(x, (tf.Tensor, tf.Variable)):
            return x.numpy()
        else:
            return np.array(x)
    elif get_backend() == "pytorch":
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().numpy()
        else:
            return np.array(x)
    else:
        return np.array(x)


def to_tensor(x: TensorLike) -> BackendTensor:
    """Make x a tensor if not already."""
    # Get numpy data if pandas
    if isinstance(x, pd.DataFrame):
        x = x.values
    elif isinstance(x, pd.Series):
        x = x.to_frame().values

    # Convert to backend tensor
    if get_backend() == "pytorch":
        import torch

        if isinstance(x, torch.Tensor):
            return x
        else:
            return torch.tensor(x)
    else:
        return x  # TensorFlow auto-converts numpy arrays to tensors


def to_default_dtype(x: TensorLike) -> TensorLike:
    """Cast a tensor to the default backend datatype."""
    if get_backend() == "pytorch":
        import torch

        if isinstance(x, torch.Tensor):
            return x.to(get_datatype())
        else:
            return torch.tensor(x).to(get_datatype())
    else:
        import tensorflow as tf

        return tf.cast(x, get_datatype())


def make_input_tensor(
    fn: Callable[Concatenate[BackendTensor, P], BackendTensor],
) -> Callable[Concatenate[TensorLike, P], BackendTensor]:
    """Decorator to cast the first argument to a function to the default backend datatype."""

    def tensor_fn(*args, **kwargs) -> BackendTensor:
        return fn(to_tensor(args[0]), *args[1:], **kwargs)

    return tensor_fn
