"""
The utils.ops module contains operations which run using the current backend.

* :func:`.kl_divergence`
* :func:`.expand_dims`
* :func:`.squeeze`
* :func:`.ones`
* :func:`.zeros`
* :func:`.full`
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

----------

"""


__all__ = [
    "kl_divergence",
    "expand_dims",
    "squeeze",
    "ones",
    "zeros",
    "full",
    "eye",
    "sum",
    "prod",
    "mean",
    "std",
    "round",
    "abs",
    "square",
    "sqrt",
    "exp",
    "relu",
    "softplus",
    "sigmoid",
    "gather",
    "cat",
    "additive_logistic_transform",
    "insert_col_of",
    "new_variable",
    "log_cholesky_transform",
]


from probflow.utils.base import BaseDistribution
from probflow.utils.casting import make_input_tensor, to_tensor
from probflow.utils.settings import get_backend, get_datatype


def kl_divergence(P, Q):
    """Compute the Kullback–Leibler divergence between two distributions.

    Parameters
    ----------
    P : |tfp.Distribution| or |torch.Distribution|
        The first distribution
    Q : |tfp.Distribution| or |torch.Distribution|
        The second distribution

    Returns
    -------
    kld : Tensor
        The Kullback–Leibler divergence between P and Q (KL(P || Q))
    """

    # Get the backend distribution if needed
    if isinstance(P, BaseDistribution):
        P = P()
    if isinstance(Q, BaseDistribution):
        Q = Q()

    # Compute KL divergence with the backend
    if get_backend() == "pytorch":
        import torch

        return torch.distributions.kl.kl_divergence(P, Q)
    else:
        import tensorflow_probability as tfp

        return tfp.distributions.kl_divergence(P, Q)


@make_input_tensor
def expand_dims(val, axis):
    """Add a singular dimension to a Tensor"""
    if axis is None:
        return val
    if get_backend() == "pytorch":
        import torch

        return torch.unsqueeze(val, axis)
    else:
        import tensorflow as tf

        return tf.expand_dims(val, axis)


@make_input_tensor
def squeeze(val):
    """Remove singleton dimensions"""
    if get_backend() == "pytorch":
        import torch

        return torch.squeeze(val)
    else:
        import tensorflow as tf

        return tf.squeeze(val)


def ones(shape):
    """Tensor full of ones."""
    if get_backend() == "pytorch":
        import torch

        return torch.ones(shape, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.ones(shape, dtype=get_datatype())


def zeros(shape):
    """Tensor full of zeros."""
    if get_backend() == "pytorch":
        import torch

        return torch.zeros(shape, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.zeros(shape, dtype=get_datatype())


def full(shape, value):
    """Tensor full of some value."""
    if get_backend() == "pytorch":
        import torch

        return torch.full(shape, value, dtype=get_datatype())
    else:
        import tensorflow as tf

        return tf.cast(tf.fill(shape, value), dtype=get_datatype())


def eye(dims):
    """Identity matrix."""
    if get_backend() == "pytorch":
        import torch

        return torch.eye(dims)
    else:
        import tensorflow as tf

        return tf.eye(dims, dtype=get_datatype())


def sum(val, axis=-1):
    """The sum."""
    if get_backend() == "pytorch":
        import torch

        if axis is None:
            return torch.sum(val)
        else:
            return torch.sum(val, axis)
    else:
        import tensorflow as tf

        return tf.reduce_sum(val, axis=axis)


def prod(val, axis=-1):
    """The product."""
    if get_backend() == "pytorch":
        import torch

        return torch.prod(val, dim=axis)
    else:
        import tensorflow as tf

        return tf.reduce_prod(val, axis=axis)


def mean(val, axis=-1):
    """The mean."""
    if get_backend() == "pytorch":
        import torch

        return torch.mean(val, dim=axis)
    else:
        import tensorflow as tf

        return tf.reduce_mean(val, axis=axis)


def std(val, axis=-1):
    """The uncorrected sample standard deviation."""
    if get_backend() == "pytorch":
        import torch

        return torch.std(val, dim=axis)
    else:
        import tensorflow as tf

        return tf.math.reduce_std(val, axis=axis)


def round(val):
    """Round to the closest integer"""
    if get_backend() == "pytorch":
        import torch

        return torch.round(val)
    else:
        import tensorflow as tf

        return tf.math.round(val)


def abs(val):
    """Absolute value"""
    if get_backend() == "pytorch":
        import torch

        return torch.abs(val)
    else:
        import tensorflow as tf

        return tf.math.abs(val)


def square(val):
    """Power of 2"""
    if get_backend() == "pytorch":
        import torch

        return val ** 2
    else:
        import tensorflow as tf

        return tf.math.square(val)


def sqrt(val):
    """The square root."""
    if get_backend() == "pytorch":
        import torch

        return torch.sqrt(val)
    else:
        import tensorflow as tf

        return tf.sqrt(val)


def exp(val):
    """The natural exponent."""
    if get_backend() == "pytorch":
        import torch

        return torch.exp(val)
    else:
        import tensorflow as tf

        return tf.exp(val)


def relu(val):
    """Linear rectification."""
    if get_backend() == "pytorch":
        import torch

        return torch.nn.ReLU()(val)
    else:
        import tensorflow as tf

        return tf.nn.relu(val)


def softplus(val):
    """Linear rectification."""
    if get_backend() == "pytorch":
        import torch

        return torch.nn.Softplus()(val)
    else:
        import tensorflow as tf

        return tf.math.softplus(val)


def sigmoid(val):
    """Sigmoid function."""
    if get_backend() == "pytorch":
        import torch

        return torch.nn.Sigmoid()(val)
    else:
        import tensorflow as tf

        return tf.math.sigmoid(val)


def gather(vals, inds, axis=0):
    """Gather values by index"""
    if get_backend() == "pytorch":
        import torch

        return torch.index_select(vals, axis, to_tensor(inds))
    else:
        import tensorflow as tf

        return tf.gather(vals, inds, axis=axis)


def cat(vals, axis=0):
    """Concatenate tensors"""
    if get_backend() == "pytorch":
        import torch

        return torch.cat(vals, dim=axis)
    else:
        import tensorflow as tf

        return tf.concat(vals, axis=axis)


def additive_logistic_transform(vals):
    """The additive logistic transformation"""
    if get_backend() == "pytorch":
        import torch

        ones_shape = [s for s in vals.shape[:-1]] + [1]
        exp_vals = torch.cat([torch.exp(vals), torch.ones(ones_shape)], dim=-1)
        return exp_vals / torch.sum(exp_vals, dim=-1, keepdim=True)
    else:
        import tensorflow as tf

        ones_shape = tf.concat([vals.shape[:-1], [1]], axis=-1)
        exp_vals = tf.concat([tf.exp(vals), tf.ones(ones_shape)], axis=-1)
        return exp_vals / tf.reduce_sum(exp_vals, axis=-1, keepdims=True)


def insert_col_of(vals, val):
    """Add a column of a value to the left side of a tensor"""
    if get_backend() == "pytorch":
        import torch

        shape = [s for s in vals.shape[:-1]] + [1]
        return torch.cat(
            [val * torch.ones(shape, dtype=get_datatype()), vals], dim=-1
        )
    else:
        import tensorflow as tf

        shape = tf.concat([vals.shape[:-1], [1]], axis=-1)
        return tf.concat(
            [val * tf.ones(shape, dtype=get_datatype()), vals], axis=-1
        )


def new_variable(initial_values):
    """Get a new variable with the current backend, and initialize it"""
    if get_backend() == "pytorch":
        import torch

        return torch.nn.Parameter(initial_values)
    else:
        import tensorflow as tf

        return tf.Variable(initial_values)


def log_cholesky_transform(x):
    r"""Perform the log cholesky transform on a vector of values.

    This turns a vector of :math:`\frac{N(N+1)}{2}` unconstrained values into a
    valid :math:`N \times N` covariance matrix.


    References
    ----------

    - Jose C. Pinheiro & Douglas M. Bates.  `Unconstrained Parameterizations
      for Variance-Covariance Matrices
      <https://dx.doi.org/10.1007/BF00140873>`_ *Statistics and Computing*,
      1996.
    """

    if get_backend() == "pytorch":
        import numpy as np
        import torch

        N = int((np.sqrt(1 + 8 * torch.numel(x)) - 1) / 2)
        E = torch.zeros((N, N))
        tril_ix = torch.tril_indices(row=N, col=N, offset=0)
        E[..., tril_ix[0], tril_ix[1]] = x
        E[..., range(N), range(N)] = torch.exp(torch.diagonal(E))
        return E @ torch.transpose(E, -1, -2)
    else:
        import tensorflow as tf
        import tensorflow_probability as tfp

        E = tfp.math.fill_triangular(x)
        E = tf.linalg.set_diag(E, tf.exp(tf.linalg.tensor_diag_part(E)))
        return E @ tf.transpose(E)
