.. _example_glm:

Poisson Regression (GLM)
========================

.. include:: macros.hrst


TODO: description... a generalized linear model w/ a Poisson observation distribution

TODO: math

TODO: diagram

.. code-block:: python

    from probflow import Exp, Dense, Poisson

    predictions = Exp(Dense())
    model = Poisson(predictions)
    model.fit(x, y)
