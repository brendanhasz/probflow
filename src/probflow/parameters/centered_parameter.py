from typing import List, Union

import numpy as np

import probflow.utils.ops as O
from probflow.distributions import Normal
from probflow.utils.casting import to_default_dtype, to_tensor

from .parameter import Parameter


class CenteredParameter(Parameter):
    r"""A vector of parameters centered at 0.

    Uses a QR decomposition to transform a vector of :math:`K-1` unconstrained
    parameters into a vector of :math:`K` variables centered at zero (i.e. the
    mean of the elements in the vector is 0).  It starts with a :math:`K \times
    K` matrix :math:`A` which has :math:`1`s along the diagonal and :math:`-1`s
    along the bottom - except for the bottom-right element which is :math:`0`:

    .. math::

        \mathbf{A} = \begin{bmatrix}
            1 & 0 & \dots & 0 & 0 \\
            0 & 1 & & 0 & 0 \\
            \vdots & & \ddots & & \vdots \\
            0 & 0 & \dots & 1 & 0 \\
            -1 & -1 & \dots & -1 & 0 \\
        \end{bmatrix}

    The :math:`QR` decomposition is performed on this matrix such that

    .. math::

        \mathbf{A} = \mathbf{Q} \mathbf{R}

    A vector of :math:`K-1` unconstrained variables :math:`\mathbf{u}` is then
    transformed into a vector :math:`\mathbf{v}` of :math:`K` centered
    variables by

    .. math::

        \mathbf{v} = \mathbf{Q}_{1:K, 1:K-1} \mathbf{u}

    The prior on the untransformed variables is

    .. math::

        \mathbf{u} \sim \text{Normal}(0, \frac{1}{\sqrt{1 - \frac{1}{K}}})

    Such that the effective prior on the transformed parameters works out to be

    .. math::

        \mathbf{v} \sim \text{Normal}(0, 1)

    .. admonition::  Prior is fixed!

        Note that the prior on the parameters is fixed at
        :math:`\text{Normal}(0, 1)`.  This is because the true prior is being
        placed on the untransformed parameters (see above).


    Parameters
    ----------
    d : int or list
        Length of the parameter vector, or shape of the parameter matrix.
    center_by : str {'all', 'column', 'row'}
        If ``all`` (the default), the sum of all parameters in the resulting
        vector or matrix will be 0.  If ``column``, the sum of each column will
        be 0 (but the sum across rows will not necessarily be 0).  If ``row``,
        the sum of each row will be 0 (but the sum across columns will not
        necessarily be 0).
    name : str
        Name of the parameter(s).
        Default = ``'CenteredParameter'``

    Examples
    --------

    TODO

    """

    def __init__(
        self,
        shape: Union[int, List[int]],
        center_by: str = "all",
        name="CenteredParameter",
    ):

        # Get a list representing the shape
        if isinstance(shape, int):
            shape = [shape]
        if len(shape) == 1:
            shape += [1]
        if len(shape) > 2:
            raise ValueError(
                "Only vector and matrix CenteredParameters are supported"
            )

        # Get the untransformed shape of the parameters
        if center_by == "row":
            K = shape[1]
            raw_shape = [K - 1, shape[0]]
        elif center_by == "column":
            K = shape[0]
            raw_shape = [K - 1, shape[1]]
        else:
            K = shape[0] * shape[1]
            raw_shape = [K - 1, 1]

        # Prior on the untransformed parameters
        scale = float(1.0 / np.sqrt(1 - 1.0 / K))
        prior = Normal(0, scale)

        # Precompute matrix by which we'll multiply the untransformed params
        A = np.eye(K)
        A[-1, :] = -1.0
        A[-1, -1] = 0.0
        Q, _ = np.linalg.qr(A)
        self._A_qr = to_default_dtype(to_tensor(Q[:, :-1]))

        # Transform function
        def A_qr_transform(u):
            if center_by == "row":
                return O.transpose(self._A_qr @ u)
            elif center_by == "all" and shape[1] > 1:
                new_shape = list(u.shape)  # to handle samples / n_mc > 1
                new_shape[-1] = shape[-1]
                new_shape[-2] = shape[-2]
                return O.reshape(self._A_qr @ u, new_shape)
            else:
                return self._A_qr @ u

        super().__init__(
            shape=raw_shape,
            posterior=Normal,
            prior=prior,
            transform=A_qr_transform,
            name=name,
        )
