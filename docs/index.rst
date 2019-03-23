ProbFlow
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   user_guide
   api
   todo

.. include:: macros.hrst

ProbFlow is a Python package for building Bayesian models with |TensorFlow Probability|, performing variational inference with those models, and evaluating the models' inferences.

It's very much still a work in progress.

- **Git repository:** http://github.com/brendanhasz/probflow
- **Documentation:** http://probflow.readthedocs.io
- **Bug reports:** http://github.com/brendanhasz/probflow/issues


Getting Started
---------------

|TensorFlow Probability| is a fantastic Python library for building probabilistic models.  Unfortunately, because of the flexibility it allows, TensorFlow Probability requires a lot of boilerplate code in order to fit a full Bayesian model.  **ProbFlow** takes care of all the legwork, allowing you to quickly and painlessly build, fit, and evaluate custom Bayesian models (or :ref:`ready-made <ready_made_models>` ones!) which run on top of |TensorFlow| and |TensorFlow Probability|.

With ProbFlow, the core building blocks of a Bayesian model are parameters, layers, and probability distributions (and, of course, the data input).  Layers define how parameters interact with the independent variables (the *x* data) to create the probability distribution of the observed dependent variables (the *y* values).

For example, a simple Bayesian linear regression

.. math::

    y \sim \text{Normal}(w x + b, \sigma)

can be built with ProbFlow by:

.. code-block:: python

    from probflow import Input, Parameter, Normal
    
    feature = Input()
    weight = Parameter()
    bias = Parameter()
    noise_std = ScaleParameter()
    
    predictions = weight*feature + bias
    model = Normal(predictions, noise_std)

Then, the model can be fit using variational inference in one line:

.. code-block:: python

    # x and y are Numpy vectors
    model.fit(x, y)

You can generate predictions for new data:

.. code-block:: python

    model.predict(x_test)

Compute *probabilistic* predictions for new data:

.. code-block:: python

    model.plot_predictive_distribution(x_test)

Evaluate your model's performance using metrics:

.. code-block:: python

    model.metrics(metric_list='accuracy')

Inspect the posterior distributions of your fit model's parameters:

.. code-block:: python

    model.plot_posterior()

and investigate how well your model is capturing uncertainty by examining how accurate its predictive intervals are:

.. code-block:: python

    model.pred_dist_covered(prc=95.0)

ProbFlow also provides more complex layers, such as those required for building Bayesian neural networks.  A multi-layer Bayesian neural network can be built and fit using ProbFlow in only a few lines:

.. code-block:: python

    from probflow import Sequential, Dense, ScaleParameter, Normal

    predictions = Sequential(layers=[
        Dense(units=128),
        Dense(units=64),
        Dense(units=1)
    ])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)

For convenience, ProbFlow also includes several :ref:`ready-made models <ready_made_models>` for standard tasks (such as linear regressions, logistic regressions, and multi-layer dense neural networks).  For example, the above linear regression example could have been done with much less work by using ProbFlow's ready-made LinearRegression model:

.. code-block:: python

    from probflow import LinearRegression

    model = LinearRegression()
    model.fit(x, y)

Using parameters, layers, and distributions as simple building blocks, ProbFlow allows for the painless creation of more complicated Bayesian models like :ref:`generalized linear models <examples_glm>`, :ref:`neural matrix factorization <examples_nmf>` models, and :ref:`mixed effects models <examples_mixed_effects>`.  Take a look at the :ref:`examples` section and the :ref:`user_guide` for more!


Installation
------------

Before installing ProbFlow, you'll first need to install `TensorFlow <http://www.tensorflow.org/install/>`_ and `TensorFlow Probability <http://www.tensorflow.org/probability/install>`_.

Then, you can use `pip <https://pypi.org/project/pip/>`_ to install ProbFlow itself from the GitHub source:

.. code-block:: bash
    
    pip install git+http://github.com/brendanhasz/probflow.git


Support
-------

Post bug reports, feature requests, and tutorial requests in `GitHub issues <http://github.com/brendanhasz/probflow/issues>`_.


Why the name, ProbFlow?
-----------------------

Because it's a package for probabilistic modeling, and it's built on TensorFlow.  ¯\\_(ツ)_/¯
