.. _example_glm:

Poisson Regression (GLM)
========================

.. include:: macros.hrst


TODO: description...

.. code-block:: python

    from probflow import Dense, Exp, Poisson

    predictions = Exp(Dense())
    model = Poisson(predictions)
    model.fit(x, y)
