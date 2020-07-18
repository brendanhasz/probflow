"""
The applications module contains pre-built |Models|

* :class:`.LinearRegression` - a linear regression model
* :class:`.LogisticRegression` - a Bi- or Multinomial logistic regression model
* :class:`.PoissonRegression` - a Poisson regression model
* :class:`.DenseRegression` - a multi-layer dense neural net regression model
* :class:`.DenseClassifier` - a multi-layer dense neural net classifier model

----------

"""


__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "PoissonRegression",
    "DenseRegression",
    "DenseClassifier",
]


from .dense_classifier import DenseClassifier
from .dense_regression import DenseRegression
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .poisson_regression import PoissonRegression
