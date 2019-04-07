.. _example_linear_regression:

Linear Regression
=================

.. include:: macros.hrst


TODO: intro, link to colab w/ these examples

.. contents:: Outline


Simple Linear Regression
------------------------

TODO: intro to a simple (bayesian) 1D linear regression:

.. math::

    y \sim \text{Normal}(w x + b, \sigma)

TODO: diagram

.. image:: img/examples/linear_regression_diagram.svg
   :width: 50 %
   :align: center

TODO: manually (generate some data, have link to colab)

.. code-block:: python

    from probflow import Input, Parameter, ScaleParameter, Normal

    feature = Input()
    weight = Parameter()
    bias = Parameter()
    noise_std = ScaleParameter()

    predictions = weight*feature + bias
    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: look at posteriors and model criticism etc


Multiple Linear Regression
--------------------------

TODO: intro to multiple linear regression

.. math::

    y \sim \text{Normal}(\mathbf{x}^\top \mathbf{w} + b, \sigma)

TODO: diagram

.. code-block:: python

    from probflow import Input, Parameter, ScaleParameter, Dot, Normal

    features = Input()
    weight = Parameter(shape=5)
    bias = Parameter()
    noise_std = ScaleParameter()

    predictions = Dot(features, weight) + bias
    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: how to access each individual posterior, etc


Using the Dense Layer
---------------------

TODO: with Dense (which automatically uses x as input if none is specified, default number of units is 1):

.. code-block:: python

    from probflow import Dense, ScaleParameter, Normal

    predictions = Dense(activation=None)
    noise_std = ScaleParameter()

    model = Normal(predictions, noise_std)
    model.fit(x, y)

TODO: how to access posterior elements from within the Dense layer


Using the LinearRegression Model
--------------------------------

TODO: with ready-made model:

.. code-block:: python

    from probflow import LinearRegression

    model = LinearRegression()
    model.fit(x, y)

TODO: how to access posterior elements from within the LinearRegression model

