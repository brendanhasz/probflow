"""Common already-made models.

TODO: more info...

* :func:`.LinearRegression`
* :func:`.LogisticRegression`
* :func:`.DenseNet`
* :func:`.DenseRegression`
* :func:`.DenseClassifier`

----------

"""

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'DenseNet',
    'DenseRegression',
    'DenseClassifier',
]

import tensorflow as tf

from .parameters import Parameter, ScaleParameter
from .layers import * #TODO: only import what you need
from .distributions import * #TODO: only import what you need



def LinearRegression(data=None):
    """Linear regression model.

    TODO: docs and math

    Parameters
    ----------
    data : |None| or a |Layer|
        Independent variable data to regress.

    """

    # Use default input if none specified
    if data is None:
        data = Input()

    # A linear regression
    error = ScaleParameter(name='noise_std')
    predictions = Dense(data)
    return Normal(predictions, error)



def LogisticRegression(data=None):
    """Logistic regression model.

    TODO: docs...

    Parameters
    ----------
    data : |None| or a |Layer|
        Independent variable data to classify.

    """

    # Use default input if none specified
    if data is None:
        data = Input()

    # A logistic regression
    predictions = Dense(data)
    return Bernoulli(predictions)



def PoissonRegression(data=None):
    """Poisson regression model.

    TODO: docs...

    Parameters
    ----------
    data : |None| or a |Layer|
        Independent variable data to regress.

    """

    # Use default input if none specified
    if data is None:
        data = Input()

    # A Poisson regression
    predictions = Exp(Dense(data))
    return Poisson(predictions)



def DenseNet(data=None, units=[1],  batch_norm=False, activation=tf.nn.relu):
    """Multiple dense layers in a row.

    .. admonition:: Does not include an observation distribution!

        :func:`.DenseNet` returns a |Layer|, unlike other ready-made models
        like :func:`.DenseRegression` which return a |Model|.  This means you
        cannot call :meth:`.fit` on the output of this function.  Instead, use
        the returned layer as a building block in a model which has an
        observation distribution.

    TODO: docs...

    Parameters
    ----------
    data : |None| or a |Layer|
        Independent variable data to regress.
    units : list of int
        List of the number of units per layer.
        Default = [1]
    batch_norm : bool
        Whether to use batch normalization in between :class:`.Dense` layers.
        Default = False
    activation : callable
        Activation function to apply after the linear transformation.
        Default = ``tf.nn.relu`` (rectified linear unit)
    """

    # Use default input if none specified
    if data is None:
        y_out = Input()
    else:
        y_out = data

    # Send output of each layer into the following layer
    for i, unit in enumerate(units):
        if i < (len(units)-1):
            y_out = Dense(y_out, units=unit, activation=activation)
        else:
            y_out = Dense(y_out, units=unit, activation=None)
        if batch_norm and i < (len(units)-1):
            y_out = BatchNormalization(y_out)

    return y_out



def DenseRegression(data=None, 
                    units=[1],
                    batch_norm=False,
                    activation=tf.nn.relu):
    """Regression model using a densely-connected multi-layer neural network.

    TODO: docs and math

    Parameters
    ----------
    data : |None| or a |Layer|
        Independent variable data to regress.
    units : list of int
        List of the number of units per layer.
        Default = [1]
    batch_norm : bool
        Whether to use batch normalization in between :class:`.Dense` layers.
        Default = False
    activation : callable
        Activation function to apply after the linear transformation.
        Default = ``tf.nn.relu`` (rectified linear unit)
    """
    error = ScaleParameter()
    predictions = DenseNet(data, units=units, 
                           batch_norm=batch_norm,
                           activation=activation)
    return Normal(predictions, error)



def DenseClassifier(data=None,
                    units=[1],
                    batch_norm=False,
                    activation=tf.nn.relu):
    """Classifier model using a densely-connected multi-layer neural network.

    TODO: docs and math
    
    Parameters
    ----------
    data : |None| or a |Layer|
        Independent variable data to classify.
    units : list of int
        List of the number of units per layer.
        Default = [1]
    batch_norm : bool
        Whether to use batch normalization in between :class:`.Dense` layers.
        Default = False
    activation : callable
        Activation function to apply after the linear transformation.
        Default = ``tf.nn.relu`` (rectified linear unit)
    """
    predictions = DenseNet(data, units=units,
                           batch_norm=batch_norm,
                           activation=activation)
    return Bernoulli(predictions)
