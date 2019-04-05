.. _example_batch_normalization:

Batch Normalization
===================

.. include:: macros.hrst


TODO

Batch normalization can be performed using the :class:`.BatchNormalization` class.  For example, to add batch normalization to the dense neural network from the previous section:

.. code-block:: python

    from probflow import BatchNormalization, Sequential, Dense, ScaleParameter, Normal

    predictions = Sequential(layers=[
        Dense(units=128),
        BatchNormalization(),
        Dense(units=64),
        BatchNormalization(),
        Dense(units=1)
    ])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)