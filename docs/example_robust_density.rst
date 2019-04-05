.. _example_robust_density:

Robust Density Network
======================

.. include:: macros.hrst


TODO: dual-module net which estimates predictions and uncertainty separately, and uses a t-dist for the observation dist

.. code-block:: python

    predictions = DenseNet(units=[128, 64, 32, 1])
    noise_std = DenseNet(units=[128, 64, 32, 1])
    model = Cauchy(predictions, noise_std)
    model.fit(x, y)