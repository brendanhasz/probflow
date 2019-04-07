.. _example_batch_normalization:

Batch Normalization
===================

.. include:: macros.hrst


TODO: intro, math, diagram


Batch normalization can be performed using the :class:`.BatchNormalization` class.  For example, to add batch normalization to the dense neural network from the `previous section <example_fully_connected>`_:

.. code-block:: python

    from probflow import Sequential, Dense, BatchNormalization, ScaleParameter, Normal

    predictions = Sequential(layers=[
        Dense(units=128),
        BatchNormalization(),
        Dense(units=64),
        BatchNormalization(),
        Dense(units=1, activation=None)
    ])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)


TODO: the :class:`.DenseNet`, :class:`.DenseRegression`, and :class:`.DenseClassifier` models take a ``batch_norm`` keyword argument which specifies whether to insert batch normalization layers between each dense layer.  The network with batch normalization in the example above could have been created using :class:`.DenseNet`:

.. code-block:: python

    from probflow import DenseNet, ScaleParameter, Normal

    predictions = DenseNet(units=[128, 64, 1], batch_norm=True)
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)

This inserts a batch normalization layer after each dense layer, except the last layer.

:class:`.DenseRegression` and :class:`.DenseClassifier` work in the same way:

.. code-block:: python

    from probflow import DenseRegression

    model = DenseRegression(units=[128, 64, 1], batch_norm=True)
    model.fit(x, y)

TODO: DenseClassifier (adds a Bernoulli observation dist):

.. code-block:: python

    from probflow import DenseClassifier

    model = DenseClassifier(units=[128, 64, 1], batch_norm=True)
    model.fit(x, y)
