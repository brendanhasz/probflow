Examples
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

TODO: intro

.. contents:: Outline

Linear Regression
-----------------

TODO: 1-variable, then w/ Dense, then w/ ready-made model

TODO: manually:

.. code-block:: python

    from probflow import Input, Variable, Normal

    feature = Input()
    weight = Variable()
    bias = Variable()
    noise_std = Variable(lb=0)

    predictions = weight*feature + bias
    model = Normal(predictions, noise_std)
    model.fit(x,y)

TODO: look at posteriors and model criticism etc


TODO: with Dense (which automatically uses x as input if none is specified):

.. code-block:: python

    from probflow import Dense, Variable, Normal

    predictions = Dense()
    noise_std = Variable(lb=0)

    model = Normal(predictions, noise_std)
    model.fit(x,y)

TODO: how to access posterior elements from within the Dense layer

TODO: with ready-made model:

.. code-block:: python

    from probflow import LinearRegression

    model = LinearRegression
    model.fit(x,y)

TODO: how to access posterior elements from within the LinearRegression model


Logistic Regression
-------------------

TODO: same idea as above just w/ predictions as input to logits for a bernoulli dist.  First do 1-variable, then w/ Dense, then w/ ready-made model


Densely-connected Neural Network
--------------------------------

TODO: manually w/ Variable s,

TODO: then w/ Sequential, 

.. code-block:: python

    from probflow import Sequential, Dense, Variable, Normal

    predictions = Sequential([
        Dense(128),
        Dense(64),
        Dense(1)
    ])
    noise_std = Variable(lb=0)
    model = Normal(predictions, noise_std)
    model.fit(x,y)

TODO: then w/ DenseRegression or DenseClassifier. (automatically sets the size of the last layer by looking @ size of input `y`, or the num unique elements of it for DenseClassifier)

.. code-block:: python

    from probflow import Sequential, Dense, Variable, Normal

    model = DenseRegression([128, 64])
    model.fit(x,y)


Robust Dual-module Neural Network
---------------------------------

TODO: dual-module net which estimates predictions and uncertainty separately, and uses a t-dist for the observation dist

.. code-block:: python

    predictions = DenseRegression([128, 64, 32])
    noise_std = DenseRegression([128, 64, 32])
    model = Cauchy(predictions, noise_std)
    model.fit(x,y)

Variational Autoencoder
-----------------------

TODO: w/ Dense, then w/ DenseAutoencoderRegression


Neural Matrix Factorization
---------------------------

TODO: w/ Dense and Embedding layers, then w/ NeuralMatrixFactorization

