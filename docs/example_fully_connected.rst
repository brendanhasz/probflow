.. _example_fully_connected:

Fully-connected Neural Network
==============================

.. include:: macros.hrst


TODO: intro, link to colab w/ these examples

.. contents:: Outline


Manually
--------

TODO: manually input -> 128 units -> 64 units -> 1 unit w/ no activation -> normal observation dist

TODO: math

TODO: diagram

TODO: this is a lot to do manually, below we'll see how to do it much more easily...

.. code-block:: python

    from probflow import Input, Parameter, ScaleParameter
    from probflow import Relu, Matmul, Dot
    from probflow import Normal

    # Input (D dimensions)
    features = Input()

    # First layer
    weights1 = Parameter(shape=[D, 128])
    bias1 = Parameter(shape=128)
    layer1 = Relu(Matmul(features, weights1) + bias1)

    # Second layer
    weights2 = Parameter(shape=[128, 64])
    bias2 = Parameter(shape=64)
    layer2 = Relu(Matmul(layer1, weights2) + bias2)

    # Last layer
    weights3 = Parameter(shape=64)
    bias3 = Parameter()
    predictions = Dot(layer2, weights3) + bias3

    # Observation distribution
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)


Using the Dense Layer
---------------------

TODO: the Dense layer automatically handles creating the parameters, performing the matrix multiplications and a additions, and applying the activation functions for you:

.. code-block:: python

    from probflow import Dense, ScaleParameter, Normal

    layer1 = Dense(units=128)
    layer2 = Dense(layer1, units=64)
    predictions = Dense(layer2, units=1, activation=None)
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)


Using the Sequential Layer
--------------------------

TODO: the Sequential layer takes a list of layers and pipes the output of each into the input of the next

.. code-block:: python

    from probflow import Sequential, Dense, ScaleParameter, Normal

    predictions = Sequential(layers=[
        Dense(units=128),
        Dense(units=64),
        Dense(units=1, activation=None)
    ])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)


Using the DenseNet Model
------------------------

TODO: the DenseNet model automatically creates sequential dense layers, but NOT an observation distribution, default is relu activation but no activation for last layer

.. code-block:: python

    from probflow import DenseNet, ScaleParameter, Normal

    predictions = DenseNet(units=[128, 64, 1])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)


Using the DenseRegression and DenseClassifier Models
----------------------------------------------------

TODO: easiest of all, the DenseRegression model adds a Normal observation dist to a DenseNet

.. code-block:: python

    from probflow import DenseRegression

    model = DenseRegression(units=[128, 64, 1])
    model.fit(x, y)

TODO: DenseClassifier (adds a Bernoulli observation dist):

.. code-block:: python

    from probflow import DenseClassifier

    model = DenseClassifier(units=[128, 64, 1])
    model.fit(x, y)
