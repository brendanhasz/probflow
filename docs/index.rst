ProbFlow
========

|Build Badge|  |Coverage Badge|  |Docs Badge|

.. |Docs Badge| image:: https://readthedocs.org/projects/probflow/badge/
    :alt: Documentation Status
    :scale: 100%
    :target: http://probflow.readthedocs.io

.. |Build Badge| image:: https://travis-ci.com/brendanhasz/probflow.svg?branch=v2
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.com/brendanhasz/probflow

.. |Coverage Badge| image:: https://codecov.io/gh/brendanhasz/probflow/branch/v2/graph/badge.svg
    :alt: Build Status
    :scale: 100%
    :target: https://codecov.io/gh/brendanhasz/probflow

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   user_guide
   examples
   api
   todo

.. include:: macros.hrst

ProbFlow is a Python package for building probabilistic Bayesian models with |TensorFlow 2.0| or |PyTorch|, performing variational inference with those models, and evaluating the models' inferences.  It provides both high-level |Modules| for building Bayesian neural networks, as well as low-level |Parameters| and |Distributions| for constructing custom Bayesian models.

It's very much still a work in progress.

- **Git repository:** http://github.com/brendanhasz/probflow
- **Documentation:** http://probflow.readthedocs.io
- **Bug reports:** http://github.com/brendanhasz/probflow/issues


Getting Started
---------------

**ProbFlow** allows you to quickly and painlessly build, fit, and evaluate custom Bayesian models (or :ref:`ready-made <ug_applications>` ones!) which run on top of either |TensorFlow| and |TensorFlow Probability| or |PyTorch|.

With ProbFlow, the core building blocks of a Bayesian model are parameters, probability distributions, and modules (and, of course, the input data).  Parameters define how the independent variables (the features) predict the probability distribution of the dependent variables (the target).

For example, a simple Bayesian linear regression

.. math::

    y \sim \text{Normal}(w x + b, \sigma)

can be built by creating a ProbFlow Model object:

.. code-block:: python3

    import probflow as pf

    class LinearRegression(pf.ContinuousModel):

        def __init__(self):
            self.weight = pf.Parameter(name='weight')
            self.bias = pf.Parameter(name='bias')
            self.std = pf.ScaleParameter(name='sigma')

        def __call__(self, x):
            return pf.Normal(x*self.weight()+self.bias(), self.std())
    
    model = LinearRegression()

Then, the model can be fit using variational inference, in *one line*:

.. code-block:: python3

    # x and y are Numpy arrays or pandas DataFrame/Series
    model.fit(x, y)

You can generate predictions for new data:

.. code-block:: python3

    # x_test is a Numpy array or pandas DataFrame
    model.predict(x_test)

Compute *probabilistic* predictions for new data, with 95% confidence intervals:

.. code-block:: python3

    model.pred_dist_plot(x_test, ci=0.95)

.. image:: img/readme/pred_dist.svg
   :width: 90 %
   :align: center

Evaluate your model's performance using various metrics:

.. code-block:: python3

    model.metric('mse', x_test, y_test)

Inspect the posterior distributions of your fit model's parameters, with 95% confidence intervals:

.. code-block:: python3

    model.posterior_plot(ci=0.95)

.. image:: img/readme/posteriors.svg
   :width: 90 %
   :align: center

Investigate how well your model is capturing uncertainty by examining how accurate its predictive intervals are:

.. code-block:: python3

    model.pred_dist_coverage(ci=0.95)

and diagnose *where* your model is having problems capturing uncertainty:

.. code-block:: python3

    model.coverage_by(ci=0.95)

.. image:: img/readme/coverage.svg
   :width: 90 %
   :align: center

ProbFlow also provides more complex layers, such as those required for building Bayesian neural networks.  Also, ProbFlow lets you mix and match ProbFlow objects with TensorFlow (or PyTorch!) objects and operations.  For example, a multi-layer Bayesian neural network can be built and fit using ProbFlow in only a few lines:

.. tabs::

    .. group-tab:: TensorFlow
        
        .. code-block:: python3

            import tensorflow as tf

            class DenseRegression(pf.ContinuousModel):

                def __init__(self, input_dims):
                    self.net = pf.Sequential([
                        pf.Dense(input_dims, 128),
                        tf.nn.relu,
                        pf.Dense(128, 64),
                        tf.nn.relu,
                        pf.Dense(64, 1),
                    ])
                    self.std = pf.ScaleParameter(name='std')

                def __call__(self, x):
                    return pf.Normal(self.net(x), self.std())
            
            model = DenseRegression()
            model.fit(x, y)

    .. group-tab:: PyTorch
        
        .. code-block:: python3

            import torch

            class DenseRegression(pf.ContinuousModel):

                def __init__(self, input_dims):
                    self.net = pf.Sequential([
                        pf.Dense(input_dims, 128),
                        torch.nn.ReLU(),
                        pf.Dense(128, 64),
                        torch.nn.ReLU(),
                        pf.Dense(64, 1),
                    ])
                    self.std = pf.ScaleParameter(name='std')

                def __call__(self, x):
                    return pf.Normal(self.net(x), self.std())
            
            model = DenseRegression()
            model.fit(x, y)

For convenience, ProbFlow also includes several :ref:`pre-built models <ug_applications>` for standard tasks (such as linear regressions, logistic regressions, and multi-layer dense neural networks).  For example, the above linear regression example could have been done with much less work by using ProbFlow's ready-made :class:`LinearRegression <probflow.applications.LinearRegression>` model:

.. code-block:: python3

    model = pf.LinearRegression(x.shape[1])
    model.fit(x, y)

And the multi-layer Bayesian neural net could have been made more easily by using ProbFlow's ready-made :class:`DenseRegression <probflow.applications.DenseRegression>` model:

.. code-block:: python3

    model = pf.DenseRegression([x.shape[1], 128, 64, 1])
    model.fit(x, y)

Using parameters and distributions as simple building blocks, ProbFlow allows for the painless creation of more complicated Bayesian models like :ref:`generalized linear models <example_glm>`, :ref:`neural matrix factorization <example_nmf>` models, and :ref:`Gaussian mixture models <example_gmm>`.  Take a look at the :ref:`examples` section and the :ref:`user_guide` for more!


Installation
------------

Before installing ProbFlow, you'll first need to install either `PyTorch <https://pytorch.org/>`_, or `TensorFlow 2.0 <https://www.tensorflow.org/install/pip>`_ and `TensorFlow Probability <http://www.tensorflow.org/probability/install>`_.  Note that currently you'll need the nightly build of TFP to work with TF 2.0.  PyTorch, TensorFlow, and TensorFlow Probability are not included in ProbFlow's `requirements.txt` file, so that you can choose which you want to use (and whether to use the GPU or CPU versions).

Then, you can install ProbFlow itself from the GitHub source:

.. code-block:: bash
    
    pip install git+http://github.com/brendanhasz/probflow.git


Version 1 vs 2
--------------

The latest version of ProbFlow (version 2) was built to work with eager execution in TensorFlow 2.x and PyTorch.  `Version 1 <https://github.com/brendanhasz/probflow/releases/tag/v1.0>`_ does not work with eager execution, and only works with TensorFlow 1.x (and not PyTorch).  The v2 interface is significantly different from v1, based on a subclassing API instead of the more declarative API of v1.  I won't be supporting v1 moving forward, but if you want to install ProbFlow 1.0:

.. code-block:: bash
    
    pip install git+http://github.com/brendanhasz/probflow.git@v1.0


Support
-------

Post bug reports, feature requests, and tutorial requests in `GitHub issues <http://github.com/brendanhasz/probflow/issues>`_.


Contributing
------------

`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_ are totally welcome!  Any contribution would be appreciated, from things as minor as pointing out typos to things as major as writing new applications and distributions.


Why the name, ProbFlow?
-----------------------

Because it's a package for probabilistic modeling, and it was built on TensorFlow.  ¯\\_(ツ)_/¯
