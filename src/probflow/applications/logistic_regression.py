import probflow.utils.ops as O
from probflow.distributions import Categorical
from probflow.models import CategoricalModel
from probflow.parameters import Parameter
from probflow.utils.casting import to_tensor


class LogisticRegression(CategoricalModel):
    r"""A logistic regression

    TODO: explain, math, diagram, examples, etc

    TODO: set k>2 for a Multinomial logistic regression

    Parameters
    ----------
    d : int
        Dimensionality of the independent variable (number of features)
    k : int
        Number of classes of the dependent variable

    Attributes
    ----------
    weights : :class:`.Parameter`
        Regression weights
    bias : :class:`.Parameter`
        Regression intercept
    """

    def __init__(self, d: int, k: int = 2):
        self.weights = Parameter([d, k - 1], name="weights")
        self.bias = Parameter([1, k - 1], name="bias")

    def __call__(self, x):
        x = to_tensor(x)
        return Categorical(
            O.insert_col_of(x @ self.weights() + self.bias(), 0)
        )
