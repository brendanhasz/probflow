"""
The utils.ops module contains operations which run using the current backend.

* :func:`.kl_divergence`
* :func:`.expand_dims`
* :func:`.squeeze`
* :func:`.ones`
* :func:`.zeros`
* :func:`.full`
* :func:`.randn`
* :func:`.rand_rademacher`
* :func:`.shape`
* :func:`.eye`
* :func:`.sum`
* :func:`.prod`
* :func:`.mean`
* :func:`.std`
* :func:`.round`
* :func:`.abs`
* :func:`.square`
* :func:`.sqrt`
* :func:`.exp`
* :func:`.relu`
* :func:`.softplus`
* :func:`.sigmoid`
* :func:`.gather`
* :func:`.cat`
* :func:`.additive_logistic_transform`
* :func:`.insert_col_of`
* :func:`.new_variable`
* :func:`.log_cholesky_transform`
* :func:`.copy_tensor`

----------

"""

__all__ = [
    "abs",
    "additive_logistic_transform",
    "cat",
    "copy_tensor",
    "exp",
    "expand_dims",
    "eye",
    "full",
    "gather",
    "insert_col_of",
    "kl_divergence",
    "log_cholesky_transform",
    "mean",
    "new_variable",
    "ones",
    "prod",
    "rand_rademacher",
    "randn",
    "relu",
    "round",
    "shape",
    "sigmoid",
    "softplus",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "sum",
    "zeros",
]

from collections.abc import Sequence

from probflow.utils.base import BaseDistribution
from probflow.utils.casting import make_input_tensor, to_tensor
from probflow.utils.settings import ProbflowBackend, get_backend, get_datatype
from probflow.utils.typing import (
    BackendDistribution,
    BackendTensor,
    BackendVariable,
    ScalarLike,
    TensorLike,
)


def kl_divergence(
    P: BaseDistribution | BackendDistribution,
    Q: BaseDistribution | BackendDistribution,
) -> ScalarLike:
    """Compute the Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    P : |Distribution| or backend distribution
        The first distribution
    Q : |Distribution| or backend distribution
        The second distribution

    Returns
    -------
    kld : Tensor
        The batchwise Kullback–Leibler divergence between P and Q (KL(P || Q)).
        For Wishart distributions, KL divergence is computed using the analytic formula for TensorFlow, PyTorch, and JAX.
    """
    # Get the backend distribution if needed
    if isinstance(P, BaseDistribution):
        P = P()
    if isinstance(Q, BaseDistribution):
        Q = Q()

    # Compute KL divergence with the backend
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        if isinstance(P, torch.distributions.Wishart) and isinstance(
            Q, torch.distributions.Wishart
        ):
            return _wishart_kl_pytorch(P, Q)
        return torch.distributions.kl.kl_divergence(P, Q)
    elif get_backend() == ProbflowBackend.JAX:
        from tensorflow_probability.substrates import jax as tfp_jax

        if isinstance(P, tfp_jax.distributions.WishartTriL) and isinstance(
            Q, tfp_jax.distributions.WishartTriL
        ):
            return _wishart_kl_jax(P, Q)
        return tfp_jax.distributions.kl_divergence(P, Q)
    else:
        import tensorflow_probability as tfp

        if isinstance(P, tfp.distributions.WishartTriL) and isinstance(
            Q, tfp.distributions.WishartTriL
        ):
            return _wishart_kl_tensorflow(P, Q)
        return tfp.distributions.kl_divergence(P, Q)


def _wishart_kl_pytorch(
    P: BackendDistribution, Q: BackendDistribution
) -> BackendTensor:
    """Calculate KL(P || Q) between two Wishart distributions, for PyTorch."""
    import torch

    if P.event_shape != Q.event_shape:
        raise ValueError(
            "Wishart distributions must have the same event shape"
        )
    d = P.event_shape[-1]
    L_p, L_q = P.scale_tril, Q.scale_tril
    logdet_p = 2 * L_p.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    logdet_q = 2 * L_q.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    solved = torch.linalg.solve_triangular(L_q, L_p, upper=False)
    trace = solved.square().sum(dim=(-2, -1))
    offsets = torch.arange(d, dtype=P.df.dtype, device=P.df.device)
    a = (P.df.unsqueeze(-1) - offsets) / 2
    b = (Q.df.unsqueeze(-1) - offsets) / 2
    return (
        0.5 * Q.df * (logdet_q - logdet_p)
        + 0.5 * P.df * (trace - d)
        + (torch.lgamma(b) - torch.lgamma(a)).sum(-1)
        + 0.5 * (P.df - Q.df) * torch.digamma(a).sum(-1)
    )


def _wishart_kl_tensorflow(
    P: BackendDistribution, Q: BackendDistribution
) -> BackendTensor:
    """Calculate KL(P || Q) between two Wishart distributions, for TensorFlow."""
    import tensorflow as tf

    if not P.event_shape.is_compatible_with(Q.event_shape):
        raise ValueError(
            "Wishart distributions must have the same event shape"
        )
    L_p, L_q = P.scale_tril, Q.scale_tril
    d = tf.shape(L_p)[-1]
    logdet_p = 2 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(L_p)), axis=-1
    )
    logdet_q = 2 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(L_q)), axis=-1
    )
    solved = tf.linalg.triangular_solve(L_q, L_p, lower=True)
    trace = tf.reduce_sum(tf.square(solved), axis=(-2, -1))
    offsets = tf.cast(tf.range(d), P.df.dtype)
    a = (P.df[..., tf.newaxis] - offsets) / 2
    b = (Q.df[..., tf.newaxis] - offsets) / 2
    return (
        0.5 * Q.df * (logdet_q - logdet_p)
        + 0.5 * P.df * (trace - tf.cast(d, P.df.dtype))
        + tf.reduce_sum(tf.math.lgamma(b) - tf.math.lgamma(a), axis=-1)
        + 0.5 * (P.df - Q.df) * tf.reduce_sum(tf.math.digamma(a), axis=-1)
    )


def _wishart_kl_jax(
    P: BackendDistribution, Q: BackendDistribution
) -> BackendTensor:
    """Calculate KL(P || Q) between two Wishart distributions, for JAX."""
    import jax.numpy as jnp
    from jax.scipy import linalg, special

    if not P.event_shape.is_compatible_with(Q.event_shape):
        raise ValueError(
            "Wishart distributions must have the same event shape"
        )
    L_p, L_q = P.scale_tril, Q.scale_tril
    d = L_p.shape[-1]
    logdet_p = 2 * jnp.log(jnp.diagonal(L_p, axis1=-2, axis2=-1)).sum(-1)
    logdet_q = 2 * jnp.log(jnp.diagonal(L_q, axis1=-2, axis2=-1)).sum(-1)
    L_q_broadcast, L_p_broadcast = jnp.broadcast_arrays(L_q, L_p)
    solved = linalg.solve_triangular(L_q_broadcast, L_p_broadcast, lower=True)
    trace = jnp.square(solved).sum(axis=(-2, -1))
    offsets = jnp.arange(d, dtype=P.df.dtype)
    a = (P.df[..., None] - offsets) / 2
    b = (Q.df[..., None] - offsets) / 2
    return (
        0.5 * Q.df * (logdet_q - logdet_p)
        + 0.5 * P.df * (trace - d)
        + (special.gammaln(b) - special.gammaln(a)).sum(-1)
        + 0.5 * (P.df - Q.df) * special.digamma(a).sum(-1)
    )


@make_input_tensor
def expand_dims(val: BackendTensor, axis: int | None) -> BackendTensor:
    """Add a singular dimension to a Tensor."""
    if axis is None:
        return val
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.unsqueeze(val, axis)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.expand_dims(val, axis)
    else:
        import tensorflow as tf

        return tf.expand_dims(val, axis)


@make_input_tensor
def squeeze(val: BackendTensor) -> BackendTensor:
    """Remove singleton dimensions."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.squeeze(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.squeeze(val)
    else:
        import tensorflow as tf

        return tf.squeeze(val)


def ones(shape: int | Sequence[int]) -> BackendTensor:
    """Tensor full of ones."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.ones(shape, dtype=get_datatype())
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.ones(shape, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.ones(shape, dtype=get_datatype())


def zeros(shape: int | Sequence[int]) -> BackendTensor:
    """Tensor full of zeros."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.zeros(shape, dtype=get_datatype())
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.zeros(shape, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.zeros(shape, dtype=get_datatype())


def full(shape: int | Sequence[int], value: float) -> BackendTensor:
    """Tensor full of some value."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.full(shape, value, dtype=get_datatype())
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.full(shape, value, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.cast(tf.fill(shape, value), dtype=get_datatype())


def randn(shape: int | Sequence[int]) -> BackendTensor:
    """Tensor full of random values drawn from a standard normal."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.randn(shape, dtype=get_datatype())
    elif get_backend() == ProbflowBackend.JAX:
        import jax

        from probflow.utils.settings import _next_jax_key

        if isinstance(shape, int):
            shape = (shape,)
        return jax.random.normal(_next_jax_key(), shape, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.random.normal(shape, dtype=get_datatype())


def rand_rademacher(shape: int | Sequence[int]) -> BackendTensor:
    """Generate a tensor full of random -1s or 1s (i.e. drawn from a Rademacher dist)."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return 2 * torch.randint(0, 2, shape, dtype=get_datatype()) - 1
    elif get_backend() == ProbflowBackend.JAX:
        from tensorflow_probability.substrates import jax as tfp_jax

        from probflow.utils.settings import _next_jax_key

        if isinstance(shape, int):
            shape = (shape,)
        return tfp_jax.random.rademacher(shape, seed=_next_jax_key())
    else:
        import tensorflow_probability as tfp

        try:  # for older versions of tfp, fall back on older version
            return tfp.random.rademacher(shape)
        except AttributeError:  # pragma: no cover
            return tfp.python.math.random_rademacher(shape)


def shape(x: BackendTensor) -> list[int]:
    """Get a list of integers representing this tensor's shape."""
    if get_backend() == ProbflowBackend.PYTORCH:
        return [s for s in x.shape]
    else:
        return [s for s in x.shape]


def eye(dims: int) -> BackendTensor:
    """Generate an identity matrix."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.eye(dims, dtype=get_datatype())
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.eye(dims, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.eye(dims, dtype=get_datatype())


def sum(
    val: BackendTensor, axis: int | None = -1, keepdims: bool = False
) -> BackendTensor:
    """Calculate the sum."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        if axis is None:
            return torch.sum(val)
        else:
            return torch.sum(val, axis, keepdim=keepdims)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.sum(val, axis=axis, keepdims=keepdims)
    else:
        import tensorflow as tf

        return tf.reduce_sum(val, axis=axis, keepdims=keepdims)


def prod(
    val: BackendTensor, axis: int | None = -1, keepdims: bool = False
) -> BackendTensor:
    """Calculate the product."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.prod(val, dim=axis, keepdim=keepdims)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.prod(val, axis=axis, keepdims=keepdims)
    else:
        import tensorflow as tf

        return tf.reduce_prod(val, axis=axis, keepdims=keepdims)


def mean(
    val: BackendTensor, axis: int | None = -1, keepdims: bool = False
) -> BackendTensor:
    """Calculate the mean."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.mean(val, dim=axis, keepdim=keepdims)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.mean(val, axis=axis, keepdims=keepdims)
    else:
        import tensorflow as tf

        return tf.reduce_mean(val, axis=axis, keepdims=keepdims)


def std(
    val: BackendTensor, axis: int | None = -1, keepdims: bool = False
) -> BackendTensor:
    """Calculate the uncorrected sample standard deviation."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.std(val, dim=axis, keepdim=keepdims)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.std(val, axis=axis, keepdims=keepdims)
    else:
        import tensorflow as tf

        return tf.math.reduce_std(val, axis=axis, keepdims=keepdims)


def round(val: BackendTensor) -> BackendTensor:
    """Round to the closest integer."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.round(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.round(val)
    else:
        import tensorflow as tf

        return tf.math.round(val)


def abs(val: BackendTensor) -> BackendTensor:
    """Take the absolute value of a tensor."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.abs(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.abs(val)
    else:
        import tensorflow as tf

        return tf.math.abs(val)


def square(val: BackendTensor) -> BackendTensor:
    """Square a tensor."""
    if get_backend() == ProbflowBackend.PYTORCH:
        return val**2
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.square(val)
    else:
        import tensorflow as tf

        return tf.math.square(val)


def sqrt(val: BackendTensor) -> BackendTensor:
    """Apply the square root."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.sqrt(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.sqrt(val)
    else:
        import tensorflow as tf

        return tf.sqrt(val)


def exp(val: BackendTensor) -> BackendTensor:
    """Apply the natural exponent."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.exp(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.exp(jnp.asarray(val))
    else:
        import tensorflow as tf

        return tf.exp(val)


def relu(val: BackendTensor) -> BackendTensor:
    """Linear rectification."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.nn.ReLU()(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.nn
        import jax.numpy as jnp

        return jax.nn.relu(jnp.asarray(val))
    else:
        import tensorflow as tf

        return tf.nn.relu(val)


def softplus(val: BackendTensor) -> BackendTensor:
    """Linear rectification."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.nn.Softplus()(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.nn
        import jax.numpy as jnp

        return jax.nn.softplus(jnp.asarray(val))
    else:
        import tensorflow as tf

        return tf.math.softplus(val)


def sigmoid(val: BackendTensor) -> BackendTensor:
    """Sigmoid function."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.nn.Sigmoid()(val)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.nn
        import jax.numpy as jnp

        return jax.nn.sigmoid(jnp.asarray(val))
    else:
        import tensorflow as tf

        return tf.math.sigmoid(val)


def gather(
    vals: BackendTensor, inds: TensorLike, axis: int = 0
) -> BackendTensor:
    """Gather values by index."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.index_select(vals, axis, to_tensor(inds))
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.take(vals, to_tensor(inds), axis=axis)
    else:
        import tensorflow as tf

        return tf.gather(vals, inds, axis=axis)


def cat(vals: Sequence[BackendTensor], axis: int = 0) -> BackendTensor:
    """Concatenate tensors."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.cat(vals, dim=axis)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.concatenate(vals, axis=axis)
    else:
        import tensorflow as tf

        return tf.concat(vals, axis=axis)


def additive_logistic_transform(vals: BackendTensor) -> BackendTensor:
    """Apply the additive logistic transformation."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        ones_shape = [s for s in vals.shape[:-1]] + [1]
        exp_vals = torch.cat(
            [torch.exp(vals), torch.ones(ones_shape, dtype=get_datatype())],
            dim=-1,
        )
        return exp_vals / torch.sum(exp_vals, dim=-1, keepdim=True)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        vals = jnp.asarray(vals)
        ones_shape = [s for s in vals.shape[:-1]] + [1]
        exp_vals = jnp.concatenate(
            [jnp.exp(vals), jnp.ones(ones_shape, dtype=get_datatype())],
            axis=-1,
        )
        return exp_vals / jnp.sum(exp_vals, axis=-1, keepdims=True)
    else:
        import tensorflow as tf

        ones_shape = tf.concat([vals.shape[:-1], [1]], axis=-1)
        exp_vals = tf.concat(
            [tf.exp(vals), tf.ones(ones_shape, dtype=get_datatype())],
            axis=-1,
        )
        return exp_vals / tf.reduce_sum(exp_vals, axis=-1, keepdims=True)


def insert_col_of(vals: BackendTensor, val: ScalarLike) -> BackendTensor:
    """Add a column of a value to the left side of a tensor."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        shape = [s for s in vals.shape[:-1]] + [1]
        return torch.cat(
            [val * torch.ones(shape, dtype=get_datatype()), vals], dim=-1
        )
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        shape = [s for s in vals.shape[:-1]] + [1]
        return jnp.concatenate(
            [val * jnp.ones(shape, dtype=get_datatype()), vals], axis=-1
        )
    else:
        import tensorflow as tf

        shape = tf.concat([vals.shape[:-1], [1]], axis=-1)
        return tf.concat(
            [val * tf.ones(shape, dtype=get_datatype()), vals], axis=-1
        )


def new_variable(initial_values: TensorLike) -> BackendVariable:
    """Get a new variable with the current backend, and initialize it."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.nn.Parameter(initial_values)
    elif get_backend() == ProbflowBackend.JAX:
        from probflow.utils.jax_variable import JaxVariable

        return JaxVariable(initial_values)
    else:
        import tensorflow as tf

        return tf.Variable(initial_values)


def cholesky(x: BackendTensor) -> BackendTensor:
    r"""Perform the cholesky decomposition on a matrix."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.linalg.cholesky(x)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.linalg.cholesky(x)
    else:
        import tensorflow as tf

        return tf.linalg.cholesky(x)


def log_cholesky_transform(
    x: BackendTensor, output_tril: bool = False
) -> BackendTensor:
    r"""Perform the log cholesky transform on a vector of values.

    This turns a vector of :math:`\frac{N(N+1)}{2}` unconstrained values into a
    valid :math:`N \times N` covariance matrix.

    Parameters
    ----------
    x : BackendTensor
        Vector of unconstrained values of length :math:`\frac{N(N+1)}{2}`.
    output_tril : bool, default False
        If True, return the lower triangular matrix instead of the full covariance matrix.

    References
    ----------
    - Jose C. Pinheiro & Douglas M. Bates.  `Unconstrained Parameterizations
      for Variance-Covariance Matrices
      <https://dx.doi.org/10.1007/BF00140873>`_ *Statistics and Computing*,
      1996.
    """
    if get_backend() == ProbflowBackend.PYTORCH:
        import numpy as np
        import torch

        N = int((np.sqrt(1 + 8 * torch.numel(x)) - 1) / 2)
        E = torch.zeros((N, N), dtype=get_datatype())
        tril_ix = torch.tril_indices(row=N, col=N, offset=0)
        E[..., tril_ix[0], tril_ix[1]] = x
        E[..., range(N), range(N)] = torch.exp(torch.diagonal(E))
        if output_tril:
            return E
        return E @ torch.transpose(E, -1, -2)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp
        from tensorflow_probability.substrates import jax as tfp_jax

        E = tfp_jax.math.fill_triangular(x)
        E = E - jnp.diag(jnp.diagonal(E)) + jnp.diag(jnp.exp(jnp.diagonal(E)))
        if output_tril:
            return E
        return E @ jnp.swapaxes(E, -1, -2)
    else:
        import tensorflow as tf
        import tensorflow_probability as tfp

        E = tfp.math.fill_triangular(x)
        E = tf.linalg.set_diag(E, tf.exp(tf.linalg.tensor_diag_part(E)))
        if output_tril:
            return E
        return E @ tf.transpose(E)


def transpose(x: BackendTensor) -> BackendTensor:
    """Transpose a matrix or batch of matrices."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.transpose(x, -1, -2)
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.swapaxes(x, -1, -2)
    else:
        import tensorflow as tf

        perm = list(range(x.ndim))
        perm[-1] = x.ndim - 2
        perm[-2] = x.ndim - 1
        return tf.transpose(x, perm=perm)


def reshape(x: BackendTensor, new_shape: Sequence[int]) -> BackendTensor:
    """Reshape a tensor."""
    if get_backend() == ProbflowBackend.PYTORCH:
        import torch

        return torch.reshape(x, tuple(new_shape))
    elif get_backend() == ProbflowBackend.JAX:
        import jax.numpy as jnp

        return jnp.reshape(x, tuple(new_shape))
    else:
        import tensorflow as tf

        return tf.reshape(x, new_shape)


def copy_tensor(x: BackendTensor) -> BackendTensor:
    """Copy a tensor, detaching it from the gradient/backend/etc/etc."""
    if get_backend() == ProbflowBackend.PYTORCH:
        return x.detach().clone()  # type: ignore
    elif get_backend() == ProbflowBackend.JAX:
        import jax
        import jax.numpy as jnp

        return jnp.array(jax.lax.stop_gradient(jnp.asarray(x)))
    else:
        import tensorflow as tf

        return tf.identity(x)
