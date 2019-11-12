ProbFlow
========

.. meta::
    :property=og\:image: img/probflow_og_image.png
    :property=og\:image\:width: 1200
    :property=og\:image\:height: 1200

|Version Badge|  |Build Badge|  |Docs Badge|  |Coverage Badge|

.. |Version Badge| image:: https://img.shields.io/pypi/v/probflow
    :target: https://pypi.org/project/probflow/

.. |Build Badge| image:: https://travis-ci.com/brendanhasz/probflow.svg
    :target: https://travis-ci.com/brendanhasz/probflow

.. |Docs Badge| image:: https://readthedocs.org/projects/probflow/badge/
    :target: http://probflow.readthedocs.io

.. |Coverage Badge| image:: https://codecov.io/gh/brendanhasz/probflow/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/brendanhasz/probflow

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   user_guide
   examples
   api
   todo

.. include:: macros.hrst

ProbFlow is a Python package for building probabilistic Bayesian models with |TensorFlow 2.0| or |PyTorch|, performing stochastic variational inference with those models, and evaluating the models' inferences.  It provides both high-level |Modules| for building Bayesian neural networks, as well as low-level |Parameters| and |Distributions| for constructing custom Bayesian models.

It's very much still a work in progress.

- **Git repository:** http://github.com/brendanhasz/probflow
- **Documentation:** http://probflow.readthedocs.io
- **Bug reports:** http://github.com/brendanhasz/probflow/issues


Getting Started
---------------

**ProbFlow** allows you to quickly and :raw-html:`<del>painlessly</del>` less painfully build, fit, and evaluate custom Bayesian models (or :doc:`ready-made <api_applications>` ones!) which run on top of either |TensorFlow| and |TensorFlow Probability| or |PyTorch|.

With ProbFlow, the core building blocks of a Bayesian model are parameters and probability distributions (and, of course, the input data).  Parameters define how the independent variables (the features) predict the probability distribution of the dependent variables (the target).

For example, a simple Bayesian linear regression

.. math::

    y \sim \text{Normal}(w x + b, \sigma)

can be built by creating a ProbFlow |Model|:


.. tabs::

    .. group-tab:: TensorFlow
            
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

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            import probflow as pf
            import torch

            class LinearRegression(pf.ContinuousModel):

                def __init__(self):
                    self.weight = pf.Parameter(name='weight')
                    self.bias = pf.Parameter(name='bias')
                    self.std = pf.ScaleParameter(name='sigma')

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Normal(x*self.weight()+self.bias(), self.std())
            
            model = LinearRegression()


Then, the model can be fit using stochastic variational inference, in *one line*:

.. code-block:: python3

    # x and y are Numpy arrays or pandas DataFrame/Series
    model.fit(x, y)

You can generate predictions for new data:

.. code-block:: pycon

    # x_test is a Numpy array or pandas DataFrame
    >>> model.predict(x_test)
    [0.983]

Compute *probabilistic* predictions for new data, with 95% confidence intervals:

.. code-block:: python3

    model.pred_dist_plot(x_test, ci=0.95)

.. image:: img/readme/pred_dist.svg
   :width: 90 %
   :align: center

Evaluate your model's performance using various metrics:

.. code-block:: pycon

    >>> model.metric('mse', x_test, y_test)
    0.217

Inspect the posterior distributions of your fit model's parameters, with 95% confidence intervals:

.. code-block:: python3

    model.posterior_plot(ci=0.95)

.. image:: img/readme/posteriors.svg
   :width: 90 %
   :align: center

Investigate how well your model is capturing uncertainty by examining how accurate its predictive intervals are:

.. code-block:: pycon

    >>> model.pred_dist_coverage(ci=0.95)
    0.903

and diagnose *where* your model is having problems capturing uncertainty:

.. code-block:: python3

    model.coverage_by(ci=0.95)

.. image:: img/readme/coverage.svg
   :width: 90 %
   :align: center

ProbFlow also provides more complex modules, such as those required for building :ref:`Bayesian neural networks <example_fully_connected>` .  Also, you can mix ProbFlow with TensorFlow (or PyTorch!) code.  For example, a multi-layer Bayesian neural network can be built and fit using ProbFlow in only a few lines:

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
                    self.std = pf.ScaleParameter()

                def __call__(self, x):
                    return pf.Normal(self.net(x), self.std())
            
            model = DenseRegression(5)
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
                    self.std = pf.ScaleParameter()

                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Normal(self.net(x), self.std())
            
            model = DenseRegression(5)
            model.fit(x, y)
            

For convenience, ProbFlow also includes several :doc:`pre-built models </api_applications>` for standard tasks (such as linear regressions, logistic regressions, and multi-layer dense neural networks).  For example, the above linear regression example could have been done with much less work by using ProbFlow's ready-made :class:`LinearRegression <probflow.applications.LinearRegression>` model:

.. code-block:: python3

    model = pf.LinearRegression(x.shape[1])
    model.fit(x, y)

And the multi-layer Bayesian neural net could have been made even more easily by using ProbFlow's ready-made :class:`DenseRegression <probflow.applications.DenseRegression>` model:

.. code-block:: python3

    model = pf.DenseRegression([x.shape[1], 128, 64, 1])
    model.fit(x, y)

Using parameters and distributions as simple building blocks, ProbFlow allows for the painless creation of more complicated Bayesian models like :ref:`generalized linear models <example_glm>`, :ref:`neural matrix factorization <example_nmf>` models, and :ref:`Gaussian mixture models <example_gmm>`.  Take a look at the :ref:`examples` section and the :ref:`user_guide` for more!


Installation
------------

Before installing ProbFlow, you'll first need to install either `PyTorch <https://pytorch.org/>`_, or `TensorFlow 2.0 <https://www.tensorflow.org/install/pip>`_ and `TensorFlow Probability <http://www.tensorflow.org/probability/install>`_.

.. tabs::

    .. tab:: PyTorch

        .. code-block:: bash
            
            pip install torch

    .. tab:: TensorFlow CPU

        .. code-block:: bash
            
            pip install tensorflow==2.0.0 tensorflow-probability==0.8.0

    .. tab:: TensorFlow GPU

        .. code-block:: bash
            
            pip install tensorflow-gpu==2.0.0 tensorflow-probability==0.8.0


Then, you can install ProbFlow itself:

.. code-block:: bash
    
    pip install probflow


Support
-------

Post bug reports, feature requests, and tutorial requests in `GitHub issues <http://github.com/brendanhasz/probflow/issues>`_.


Contributing
------------

`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_ are totally welcome!  Any contribution would be appreciated, from things as minor as pointing out typos to things as major as writing new applications and distributions.


Why the name, ProbFlow?
-----------------------

Because it's a package for probabilistic modeling, and it was built on TensorFlow.  ¯\\_(ツ)_/¯
