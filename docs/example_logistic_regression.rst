.. _example_logistic_regression:

Logistic Regression
===================

.. include:: macros.hrst


TODO: intro, link to colab w/ these examples

.. contents:: Outline


Logistic Regression
-------------------

TODO: logistic regression...

.. math::

    y \sim \text{Bernoulli}(\mathbf{x}^\top \mathbf{w} + b)

TODO: diagram

TODO: generate some data

TODO: create the model via the subclassing api

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            import probflow as pf

            class LogisticRegression(pf.Model):

                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 1])
                    self.b = pf.Parameter()

                def __call__(self, x):
                    return pf.Bernoulli(x @ self.w() + self.b())

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            import probflow as pf
            import torch

            class LogisticRegression(pf.Model):

                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 1])
                    self.b = pf.Parameter()

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Bernoulli(x @ self.w() + self.b())

By default, the :class:`.Bernoulli` distribution treats its inputs as logits (that is, it passes the inputs through a sigmoid function to get the output class probabilities).  To force it to treat the inputs as raw probability values, use the ``probs`` keyword argument:

.. code-block:: python3

    pf.Bernoulli(probs=x @ self.w() + self.b())

TODO: initialize and fit

.. code-block:: python3

    model = LogisticRegression()
    model.fit(x, y)

TODO: look at model criticism for a categorical model


Using the Dense Layer
---------------------

TODO: can do the same thing with Dense:

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            class LogisticRegression(pf.Model):

                def __init__(self, dims):
                    self.layer = pf.Dense(dims, 1)

                def __call__(self, x):
                    return pf.Bernoulli(self.layer(x))

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            class LogisticRegression(pf.Model):

                def __init__(self, dims):
                    self.layer = pf.Dense(dims, 1)

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Bernoulli(self.layer(x))


Using the LogisticRegression Model
----------------------------------

TODO: and can just use the ready-made model:

.. code-block:: python3

    model = pf.LogisticRegression()
    model.fit(x, y)

TODO: how to access params from w/i the model
