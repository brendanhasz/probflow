import pandas as pd
import numpy as np
from probflow.utils.typing import TensorLike
from probflow.utils.casting import to_numpy
from probflow.utils.validation import ensure_tensor_like


def get_ndims(x: TensorLike, name: str = "") -> int:
    """Get the number of dimensions of any tensor-like object"""
    ensure_tensor_like(x, name)
    if isinstance(x, (int, float)):
        return 1
    if isinstance(x, list):
        return int(to_numpy(x).ndim)
    if isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
        return int(x.ndim)
    else:
        return int(x.ndim)