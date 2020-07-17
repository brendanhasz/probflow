from typing import List, Callable, Union

import probflow.utils.ops as O
from probflow.utils.casting import to_tensor
from probflow.parameters import Parameter
from probflow.parameters import ScaleParameter
from probflow.distributions import Normal
from probflow.models import ContinuousModel


class LinearRegression(ContinuousModel):
    r"""A multiple linear regression

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : int
        Dimensionality of the independent variable (number of features)
    heteroscedastic : bool
        Whether to model a change in noise as a function of :math:`\mathbf{x}`
        (if ``heteroscedastic=True``), or not (if ``heteroscedastic=False``,
        the default).

    Attributes
    ----------
    weights : :class:`.Parameter`
        Regression weights
    bias : :class:`.Parameter`
        Regression intercept
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, d: int, heteroscedastic: bool = False):
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.weights = Parameter([d, 2], name="weights")
            self.bias = Parameter([1, 2], name="bias")
        else:
            self.weights = Parameter([d, 1], name="weights")
            self.bias = Parameter([1, 1], name="bias")
            self.std = ScaleParameter([1, 1], name="std")

    def __call__(self, x):
        x = to_tensor(x)
        if self.heteroscedastic:
            p = x @ self.weights() + self.bias()
            m_preds = p[..., :, 0:1]
            s_preds = O.exp(p[..., :, 1:2])
            return Normal(m_preds, s_preds)
        else:
            return Normal(x @ self.weights() + self.bias(), self.std())
