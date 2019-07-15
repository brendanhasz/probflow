"""Ready-made models

The applications module contains pre-built |Models| and |Modules|.

"""


__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'PoissonRegression',
    'DenseNetwork',
    'DenseRegression',
    'DenseClassifier',
]


import probflow.core.ops as O
from probflow.parameters import Parameter
from probflow.parameters import ScaleParameter
from probflow.distributions import Normal
from probflow.distributions import Bernoulli
from probflow.distributions import Poisson
from probflow.modules import Module
from probflow.modules import Dense
from probflow.models import ContinuousModel
from probflow.models import DiscreteModel
from probflow.models import CategoricalModel



class LinearRegression(ContinuousModel):
    """TODO"""

    def __init__(self, dims):
        self.weights = Parameter([dims, 1])
        self.bias = Parameter()
        self.std = ScaleParameter()


    def __call__(self, x):
        return Normal(x @ self.weights() + self.bias(), self.std())



class LogisticRegression(CategoricalModel):
    """TODO"""

    def __init__(self, dims):
        self.weights = Parameter([dims, 1])
        self.bias = Parameter()


    def __call__(self, x):
        return Bernoulli(x @ self.weights() + self.bias())



class PoissonRegression(DiscreteModel):
    """TODO"""

    def __init__(self, dims):
        self.weights = Parameter([dims, 1])
        self.bias = Parameter()


    def __call__(self, x):
        return Poisson(O.exp(x @ self.weights() + self.bias()))



class DenseNetwork(Module):
    """TODO"""

    def __init__(self, dims, activation=O.relu):
        self.activations = [activation for i in range(len(dims)-2)]
        self.activations += [lambda x: x]
        self.layers = [Dense(dims[i], dims[i+1])
                       for i in range(len(dims)-1)]


    def __call__(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x



class DenseRegression(ContinuousModel):
    """TODO"""

    def __init__(self, dims):
        self.network = DenseNetwork(dims)
        self.std = ScaleParameter()


    def __call__(self, x):
        return Normal(self.network(x), self.std())



class DenseClassifier(CategoricalModel):
    """TODO"""

    def __init__(self, dims):
        self.network = DenseNetwork(dims)


    def __call__(self, x):
        return Bernoulli(self.network(x))
