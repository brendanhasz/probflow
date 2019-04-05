.. _example_fully_connected:

Fully-connected Neural Network
==============================

.. include:: macros.hrst


TODO: manually w/ Parameter s,

TODO: then w/ Sequential, 

.. code-block:: python

    from probflow import Sequential, Dense, ScaleParameter, Normal

    predictions = Sequential(layers=[
        Dense(units=128),
        Dense(units=64),
        Dense(units=1)
    ])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: then w/ DenseNet, which automatically creates sequential dense layers, but NOT the normal dist on top

.. code-block:: python

    from probflow import Sequential, Dense, ScaleParameter, Normal

    predictions = DenseNet(units=[128, 64, 1])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: then w/ DenseRegression (adds a normal dist observation dist) or DenseClassifier (adds a Bernoulli dist):

.. code-block:: python

    from probflow import DenseRegression

    model = DenseRegression(units=[128, 64, 1])
    model.fit(x, y)