from typing import Callable, List, Union

import probflow.utils.ops as O
from probflow.distributions import Normal
from probflow.models import ContinuousModel
from probflow.parameters import Parameter, ScaleParameter
from probflow.utils.casting import to_tensor


class LinearRegression(ContinuousModel):
    r"""A multiple linear regression

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : int
        Dimensionality of the independent variable (number of features)
    d_o : int
        Dimensionality of the dependent variable (number of target dimensions)
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

    def __init__(self, d: int, d_o: int = 1, heteroscedastic: bool = False):
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.d_o = d_o
            self.weights = Parameter([d, d_o * 2], name="weights")
            self.bias = Parameter([1, d_o * 2], name="bias")
        else:
            self.weights = Parameter([d, d_o], name="weights")
            self.bias = Parameter([1, d_o], name="bias")
            self.std = ScaleParameter([1, d_o], name="std")

    def __call__(self, x):
        x = to_tensor(x)
        if self.heteroscedastic:
            p = x @ self.weights() + self.bias()
            m_preds = p[..., :, : self.d_o]
            s_preds = O.exp(p[..., :, self.d_o :])
            return Normal(m_preds, s_preds)
        else:
            return Normal(x @ self.weights() + self.bias(), self.std())
