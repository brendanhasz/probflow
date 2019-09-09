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

TODO: manually (generate some data)

TODO: talk about subclassing API

.. code-block:: python

    import probflow as pf

    class SimpleLinearRegression(pf.Model):

        def __init__(self):
            self.w = pf.Parameter()
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(x*self.w()+self.b(), self.s())

TODO: initialize and fit

.. code-block:: python

    model = SimpleLinearRegression()
    model.fit(x, y)

TODO: look at posteriors and model criticism etc


Multiple Linear Regression
--------------------------

TODO: intro to multiple linear regression

.. math::

    y \sim \text{Normal}(\mathbf{x}^\top \mathbf{w} + b, \sigma)

TODO: diagram

TODO: generate some data

TODO: again build w/ subclassing API

.. code-block:: python

    class MultipleLinearRegression(pf.Model):

        def __init__(self, dims):
            self.w = pf.Parameter([dims, 1])
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(x @ self.w() + self.b(), self.s())

TODO: initialize and fit

.. code-block:: python

    model = MultipleLinearRegression()
    model.fit(x, y)

TODO: how to access each individual posterior, etc


Using the LinearRegression Model
--------------------------------

TODO: with ready-made model:

.. code-block:: python

    model = pf.LinearRegression()
    model.fit(x, y)

TODO: how to access posterior elements from within the LinearRegression model

