"""
The applications module contains pre-built |Models| and |Modules|.

* :class:`.LinearRegression` - a linear regression model
* :class:`.LogisticRegression` - a Bi- or Multinomial logistic regression model
* :class:`.PoissonRegression` - a Poisson regression model
* :class:`.DenseNetwork` - a multi-layer dense neural network module
* :class:`.DenseRegression` - a multi-layer dense neural net regression model
* :class:`.DenseClassifier` - a multi-layer dense neural net classifier model

----------

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
from probflow.utils.casting import to_tensor
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
            self.weights = Parameter([d, 2], name='weights')
            self.bias = Parameter([1, 1], name='bias')
        else:
            self.weights = Parameter([d, 1], name='weights')
            self.bias = Parameter([1, 1], name='bias')
            self.std = ScaleParameter([1, 1], name='std')


    def __call__(self, x):
        x = to_tensor(x)
        if self.heteroscedastic:
            p = x @ self.weights()
            m_preds = p[..., :, 0:1] + self.bias()
            s_preds = O.exp(p[..., :, 1:2])
            return Normal(m_preds, s_preds)
        else:
            return Normal(x @ self.weights() + self.bias(), self.std())



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
        self.weights = Parameter([d, k-1], name='weights')
        self.bias = Parameter([1, k-1], name='bias')


    def __call__(self, x):
        x = to_tensor(x)
        return Categorical(O.insert_col_of(x @ self.weights() + self.bias(), 0))



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
        self.weights = Parameter([d, 1], name='weights')
        self.bias = Parameter([1, 1], name='bias')


    def __call__(self, x):
        x = to_tensor(x)
        return Poisson(O.exp(x @ self.weights() + self.bias()))



class DenseNetwork(Module):
    r"""A multilayer dense neural network

    TODO: warning about how this is a Module not a Model

    TODO: explain, math, diagram, examples, etc

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
        x = to_tensor(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x



class DenseRegression(ContinuousModel):
    r"""A regression using a multilayer dense neural network

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        dimensionality of the dependent variable (number of dimensions of the
        target).
    heteroscedastic : bool 
        Whether to model a change in noise as a function of :math:`\mathbf{x}`
        (if ``heteroscedastic=True``), or not (if ``heteroscedastic=False``,
        the default).
    kwargs
        Additional keyword arguments are passed to :class:`.DenseNetwork`

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        mean
    std : :class:`.ScaleParameter`
        Standard deviation of the Normal observation distribution
    """

    def __init__(self, d: List[int], heteroscedastic: bool = False, **kwargs):
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            d[-1] = 2*d[-1]
            self.network = DenseNetwork(d, **kwargs)
        else:
            self.network = DenseNetwork(d, **kwargs)
            self.std = ScaleParameter([1, 1], name='std')


    def __call__(self, x):
        x = to_tensor(x)
        if self.heteroscedastic:
            p = self.network(x)
            Nd = int(p.shape[-1]/2)
            m_preds = p[..., :, 0:Nd]
            s_preds = O.exp(p[..., :, Nd:2*Nd])
            return Normal(m_preds, s_preds)
        else:
            return Normal(self.network(x), self.std())



class DenseClassifier(CategoricalModel):
    r"""A classifier which uses a multilayer dense neural network

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features), and the last element should be the
        number of classes of the target.
    kwargs
        Additional keyword arguments are passed to :class:`.DenseNetwork`

    Attributes
    ----------
    network : :class:`.DenseNetwork`
        The multilayer dense neural network which generates predictions of the
        class probabilities
    """

    def __init__(self, d: List[int], **kwargs):
        d[-1] -= 1
        self.network = DenseNetwork(d, **kwargs)


    def __call__(self, x):
        x = to_tensor(x)
        return Categorical(O.add_col_of(self.network(x), 0))
