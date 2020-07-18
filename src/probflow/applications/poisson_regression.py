import probflow.utils.ops as O
from probflow.distributions import Poisson
from probflow.models import DiscreteModel
from probflow.parameters import Parameter
from probflow.utils.casting import to_tensor


class PoissonRegression(DiscreteModel):
    r"""A Poisson regression (a type of generalized linear model)

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : int
        Dimensionality of the independent variable (number of features)

    Attributes
    ----------
    weights : :class:`.Parameter`
        Regression weights
    bias : :class:`.Parameter`
        Regression intercept
    """

    def __init__(self, d: int):
        self.weights = Parameter([d, 1], name="weights")
        self.bias = Parameter([1, 1], name="bias")

    def __call__(self, x):
        x = to_tensor(x)
        return Poisson(O.exp(x @ self.weights() + self.bias()))
