"""Ready-made models.

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



from typing import List, Callable, Union

import probflow.core.ops as O
from probflow.parameters import Parameter
from probflow.parameters import ScaleParameter
from probflow.distributions import Normal
from probflow.distributions import Bernoulli
from probflow.distributions import Categorical
from probflow.distributions import Poisson
from probflow.modules import Module
from probflow.modules import Dense
from probflow.models import ContinuousModel
from probflow.models import DiscreteModel
from probflow.models import CategoricalModel



class LinearRegression(ContinuousModel):
    r"""A multiple linear regression

    TODO: explain, math, diagram, etc

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
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, d: int):
        self.weights = Parameter([d, 1], name='weights')
        self.bias = Parameter(name='bias')
        self.std = ScaleParameter(name='std')


    def __call__(self, x):
        return Normal(x @ self.weights() + self.bias(), self.std())



class LogisticRegression(CategoricalModel):
    r"""A logistic regression

    TODO: explain, math, diagram, etc

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
        self.weights = Parameter([d, k-1], name='weights')
        self.bias = Parameter([k-1], name='bias')


    def __call__(self, x):
        return Categorical(O.add_col_of(x @ self.weights() + self.bias(), 0))



class PoissonRegression(DiscreteModel):
    r"""A Poisson regression (a type of generalized linear model)

    TODO: explain, math, diagram, etc

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
        self.weights = Parameter([d, 1], name='weights')
        self.bias = Parameter(name='bias')


    def __call__(self, x):
        return Poisson(O.exp(x @ self.weights() + self.bias()))



class DenseNetwork(Module):
    r"""A multilayer dense neural network

    TODO: warning about how this is a Module not a Model

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features).
    activation : callable
        Activation function to apply to the outputs of each layer.
        Note that the activation function will not be applied to the outputs
        of the final layer.
        Default = :math:`\max ( 0, x )`

    Attributes
    ----------
    layers : List[:class:`.Dense`]
        List of :class:`.Dense` neural network layers to be applied
    activations : List[callable]
        Activation function for each layer
    """

    def __init__(self, 
                 d: List[int], 
                 activation: Callable = O.relu,
                 name: Union[str, None] = None):
        self.activations = [activation for i in range(len(d)-2)]
        self.activations += [lambda x: x]
        name = '' if name is None else name+'_'
        names = [name+'Dense'+str(i) for i in range(len(d)-1)]
        self.layers = [Dense(d[i], d[i+1], name=names[i]) 
                       for i in range(len(d)-1)]


    def __call__(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x



class DenseRegression(ContinuousModel):
    r"""A regression using a multilayer dense neural network

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        dimensionality of the dependent variable (number of dimensions of the
        target).

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        mean
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, d: List[int]):
        self.network = DenseNetwork(d)
        self.std = ScaleParameter(name='std')


    def __call__(self, x):
        return Normal(self.network(x), self.std())



class DenseClassifier(CategoricalModel):
    r"""A classifier which uses a multilayer dense neural network

    TODO: explain, math, diagram, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        number of classes of the target.

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        class probabilities
    """

    def __init__(self, d: List[int]):
        d[-1] -= 1
        self.network = DenseNetwork(d)


    def __call__(self, x):
        return Categorical(O.add_col_of(self.network(x), 0))
