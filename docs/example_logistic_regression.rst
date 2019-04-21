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

.. code-block:: python

    from probflow import Input, Parameter, Dot, Bernoulli

    features = Input()
    weights = Parameter(shape=5)
    bias = Parameter()

    logits = Dot(features, weights) + bias
    model = Bernoulli(logits)
    model.fit(x, y)

By default, the :class:`.Bernoulli` distribution treats its inputs as logits (that is, it passes the inputs through a sigmoid function to get the output class probabilities).  To force it to treat the inputs as raw probability values, use the ``input_type`` keyword argument:

.. code-block:: python

    from probflow import Sigmoid

    probs = Sigmoid(logits)
    model = Bernoulli(probs, input_type='probs')
    model.fit(x, y)


Using the Dense Layer
---------------------

TODO: with Dense (which automatically uses x as input if none is specified, default number of units is 1):

.. code-block:: python

    from probflow import Dense, Bernoulli

    logits = Dense(activation=None)
    model = Bernoulli(logits)
    model.fit(x, y)


Using the LogisticRegression Model
----------------------------------

TODO: with ready-made model:

.. code-block:: python

    from probflow import LogisticRegression

    model = LogisticRegression()
    model.fit(x, y)

