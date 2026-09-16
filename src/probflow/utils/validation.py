"""Functions for checking that inputs have the correct type.

* :func:`.ensure_tensor_like`

"""

from typing import Any

import numpy as np
import pandas as pd

from probflow.utils.base import BaseParameter
from probflow.utils.settings import ProbflowBackend, get_backend


def ensure_tensor_like(obj: Any, name: str) -> None:
    """Determine whether an object can be cast to a Tensor."""
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
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        if not isinstance(obj, (torch.Tensor, BaseParameter)):
            raise TypeError(name + " must be Tensor-like")
    elif get_backend() == ProbflowBackend.JAX:
        import jax

        from probflow.utils.jax_variable import JaxVariable

        if not isinstance(obj, (jax.Array, JaxVariable, BaseParameter)):
            raise TypeError(name + " must be Tensor-like")
    else:
        import tensorflow as tf

        if not isinstance(obj, (tf.Tensor, tf.Variable, BaseParameter)):
            raise TypeError(name + " must be Tensor-like")


def is_backend_tensor(obj: Any) -> bool:
    """Determine whether an object is a backend Tensor."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return isinstance(obj, (torch.Tensor, torch.nn.Parameter))
    elif get_backend() == ProbflowBackend.JAX:
        import jax

        from probflow.utils.jax_variable import JaxVariable

        return isinstance(obj, (jax.Array, JaxVariable))
    else:
        import tensorflow as tf

        return isinstance(obj, (tf.Tensor, tf.Variable))


def is_backend_variable(obj: Any) -> bool:
    """Determine whether an object is a backend Variable."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return isinstance(obj, torch.nn.Parameter)
    elif get_backend() == ProbflowBackend.JAX:
        from probflow.utils.jax_variable import JaxVariable

        return isinstance(obj, JaxVariable)
    else:
        import tensorflow as tf

        return isinstance(obj, tf.Variable)


def is_backend_distribution(obj: Any) -> bool:
    """Determine whether an object is a backend distribution."""
    if get_backend() == ProbflowBackend.PYTORCH:
        # PyTorch distributions are instances of torch.distributions.Distribution
        from torch.distributions import Distribution as TorchDistribution

        return isinstance(obj, TorchDistribution)
    elif get_backend() == ProbflowBackend.JAX:
        from tensorflow_probability.substrates import jax as tfp_jax

        return isinstance(obj, tfp_jax.distributions.Distribution)
    else:
        import tensorflow_probability as tfp

        # TensorFlow Probability distributions are instances of tfp.distributions.Distribution
        return isinstance(obj, tfp.distributions.Distribution)
