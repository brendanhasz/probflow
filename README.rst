ProbFlow
========

|Docs Badge|

.. |Docs Badge| image:: https://readthedocs.org/projects/probflow/badge/
    :alt: Documentation Status
    :scale: 100%
    :target: http://probflow.readthedocs.io

ProbFlow is a Python package for building probabilistic Bayesian models with `TensorFlow Probability <http://www.tensorflow.org/probability>`_, performing variational inference with those models, and evaluating the models' inferences.  It provides both high-level layers for building Bayesian neural networks, and low-level modules for constructing custom Bayesian models.

It's very much still a work in progress.

- **Git repository:** http://github.com/brendanhasz/probflow
- **Documentation:** http://probflow.readthedocs.io
- **Bug reports:** http://github.com/brendanhasz/probflow/issues


Getting Started
---------------

**ProbFlow** allows you to quickly and painlessly build, fit, and evaluate custom Bayesian models (or `ready-made <http://probflow.readthedocs.io/en/latest/ready_made_models.html>`_ ones!) which run on top of `TensorFlow <http://www.tensorflow.org/>`_ and `TensorFlow Probability <http://www.tensorflow.org/probability>`_.

With ProbFlow, the core building blocks of a Bayesian model are parameters, layers, and probability distributions (and, of course, the input data).  Layers define how parameters interact with the independent variables (the features) to predict the probability distribution of the dependent variables (the target).

For example, a simple Bayesian linear regression

.. image:: docs/img/readme/regression_equation.svg
   :width: 30 %
   :align: center

can be built with ProbFlow by:

.. code-block:: python

    from probflow import Input, Parameter, ScaleParameter, Normal
    
    feature = Input()
    weight = Parameter()
    bias = Parameter()
    noise_std = ScaleParameter()
    
    predictions = weight*feature + bias
    model = Normal(predictions, noise_std)

Then, the model can be fit using variational inference, in *one line*:

.. code-block:: python

    # x and y are Numpy arrays or pandas DataFrame/Series
    model.fit(x, y)

You can generate predictions for new data:

.. code-block:: python

    # x_test is a Numpy array or pandas DataFrame
    model.predict(x_test)

Compute *probabilistic* predictions for new data, with 95% confidence intervals:

.. code-block:: python

    model.predictive_distribution_plot(x_test, ci=0.95)

.. image:: docs/img/readme/pred_dist.svg
   :width: 90 %
   :align: center

Evaluate your model's performance using metrics:

.. code-block:: python

    model.metrics('mse')

Inspect the posterior distributions of your fit model's parameters, with 95% confidence intervals:

.. code-block:: python

    model.posterior_plot(ci=0.95)

.. image:: docs/img/readme/posteriors.svg
   :width: 90 %
   :align: center

Investigate how well your model is capturing uncertainty by examining how accurate its predictive intervals are:

.. code-block:: python

    model.pred_dist_coverage(ci=0.95)

and diagnose *where* your model is having problems capturing uncertainty:

.. code-block:: python

    model.coverage_by(ci=0.95)

.. image:: docs/img/readme/coverage.svg
   :width: 90 %
   :align: center

ProbFlow also provides more complex layers, such as those required for building Bayesian neural networks.  A multi-layer Bayesian neural network can be built and fit using ProbFlow in only a few lines:

.. code-block:: python

    from probflow import Sequential, Dense, ScaleParameter, Normal

    predictions = Sequential(layers=[
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])
    noise_std = ScaleParameter()
    model = Normal(predictions, noise_std)
    model.fit(x, y)

For convenience, ProbFlow also includes several `ready-made models <http://probflow.readthedocs.io/en/latest/ready_made_models.html>`_ for standard tasks (such as linear regressions, logistic regressions, and multi-layer dense neural networks).  For example, the above linear regression example could have been done with much less work by using ProbFlow's ready-made LinearRegression model:

.. code-block:: python

    from probflow import LinearRegression

    model = LinearRegression()
    model.fit(x, y)

And the multi-layer Bayesian neural net could have been made more easily by using ProbFlow's ready-made DenseRegression model:

.. code-block:: python

    from probflow import DenseRegression

    model = DenseRegression(units=[128, 64, 1])
    model.fit(x, y)

Using parameters, layers, and distributions as simple building blocks, ProbFlow allows for the painless creation of more complicated Bayesian models like `generalized linear models <http://probflow.readthedocs.io/en/latest/example_glm.html>`_, `neural matrix factorization <http://probflow.readthedocs.io/en/latest/example_nmf.html>`_ models, and `mixed effects models <http://probflow.readthedocs.io/en/latest/example_multilevel.html>`_.  Take a look at the `examples <http://probflow.readthedocs.io/en/latest/examples.html>`_ section and the `user guide <http://probflow.readthedocs.io/en/latest/user_guide.html>`_ for more!


Installation
------------

Before installing ProbFlow, you'll first need to either `PyTorch <>`, or `TensorFlow <http://www.tensorflow.org/install/>`_ and `TensorFlow Probability <http://www.tensorflow.org/probability/install>`_.  PyTorch, TensorFlow, and TensorFlow Probability are not included in ProbFlow's `requirements.txt` file, so that you can choose which you want to use (and whether to use the GPU or CPU versions).

Then, you can use `pip <http://pypi.org/project/pip/>`_ to install ProbFlow itself from the GitHub source:

.. code-block::
    
    pip install git+http://github.com/brendanhasz/probflow.git


Version 1.0 vs 2.0
------------------

The latest version of ProbFlow (version 2) was built to work with eager execution in TensorFlow 2.0 and PyTorch.  Version 1 does not work with eager execution, and only works with TensorFlow 1.0.  The interface is significantly different from 1.0.  To install ProbFlow 1.0:

.. code-block::
    
    pip install git+http://github.com/brendanhasz/probflow.git@v1.0


Support
-------

Post bug reports, feature requests, and tutorial requests in `GitHub issues <http://github.com/brendanhasz/probflow/issues>`_.


Contributing
------------

`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_ are totally welcome!  Any contribution would be appreciated, from things as minor as pointing out typos to things as major as writing new layers and distributions.


Why the name, ProbFlow?
-----------------------

Because it's a package for probabilistic modeling, and it was originally built on TensorFlow.  ¯\\_(ツ)_/¯
