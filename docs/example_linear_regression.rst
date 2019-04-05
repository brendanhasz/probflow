.. _example_linear_regression:

Linear Regression
=================

.. include:: macros.hrst


TODO: explain a basic (bayesian) 1D linear regression:

.. math::

    \mathbf{y} \sim \text{Normal}(\mathbf{x} \beta_1 + \beta_0, \sigma)

TODO: diagram...

.. image:: img/examples/linear_regression_diagram.svg
   :width: 50 %
   :align: center

TODO: manually:

.. code-block:: python

    from probflow import Input, Parameter, Normal

    feature = Input()
    weight = Parameter()
    bias = Parameter()
    noise_std = ScaleParameter()

    predictions = weight*feature + bias
    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: look at posteriors and model criticism etc

TODO: multiple linear regression, posteriors, etc

TODO: with Dense (which automatically uses x as input if none is specified):

.. code-block:: python

    from probflow import Dense, Parameter, Normal

    predictions = Dense()
    noise_std = ScaleParameter()

    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: how to access posterior elements from within the Dense layer

TODO: with ready-made model:

.. code-block:: python

    from probflow import LinearRegression

    model = LinearRegression()
    model.fit(x, y)

TODO: how to access posterior elements from within the LinearRegression model

