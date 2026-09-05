"""Functions for getting tensor shapes."""

import numpy as np
import pandas as pd

from probflow.utils.casting import to_numpy
from probflow.utils.typing import TensorLike
from probflow.utils.validation import ensure_tensor_like


def get_ndims(x: TensorLike, name: str = "") -> int:
    """Get the number of dimensions of any tensor-like object."""
    ensure_tensor_like(x, name)
    if isinstance(x, (int, float)):
        return 1
    if isinstance(x, list):
        return int(to_numpy(x).ndim)
    if isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
        return int(x.ndim)
    else:
        return int(x.ndim)


def get_shape(x: TensorLike) -> tuple[int, ...]:
    """Get the shape of any tensor-like object."""
    if isinstance(x, (int, float)):
        return (1,)
    if isinstance(x, list):
        return to_numpy(x).shape  # type: ignore
    else:
        return x.shape  # type: ignore
