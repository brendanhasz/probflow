ProbFlow
========

|Version Badge|  |Build Badge|  |Docs Badge|  |Coverage Badge|

.. |Version Badge| image:: https://img.shields.io/pypi/v/probflow
    :target: https://pypi.org/project/probflow/

.. |Build Badge| image:: https://github.com/brendanhasz/probflow/workflows/tests/badge.svg
    :target: https://github.com/brendanhasz/probflow/actions?query=branch%3Amaster

.. |Docs Badge| image:: https://readthedocs.org/projects/probflow/badge/
    :target: http://probflow.readthedocs.io

.. |Coverage Badge| image:: https://codecov.io/gh/brendanhasz/probflow/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/brendanhasz/probflow


ProbFlow is a Python package for building probabilistic Bayesian models with `TensorFlow 2.0 <http://www.tensorflow.org/beta>`_ or `PyTorch <http://pytorch.org>`_, performing stochastic variational inference with those models, and evaluating the models' inferences.  It provides both high-level modules for building Bayesian neural networks, as well as low-level parameters and distributions for constructing custom Bayesian models.

It's very much still a work in progress.

- **Git repository:** http://github.com/brendanhasz/probflow
- **Documentation:** http://probflow.readthedocs.io
- **Bug reports:** http://github.com/brendanhasz/probflow/issues


Getting Started
---------------

**ProbFlow** allows you to quickly and less painfully build, fit, and evaluate custom Bayesian models (or `ready-made <http://probflow.readthedocs.io/en/latest/api_applications.html>`_ ones!) which run on top of either `TensorFlow 2.0 <http://www.tensorflow.org/beta>`_ and `TensorFlow Probability <http://www.tensorflow.org/probability>`_ or `PyTorch <http://pytorch.org>`_.

With ProbFlow, the core building blocks of a Bayesian model are parameters and probability distributions (and, of course, the input data).  Parameters define how the independent variables (the features) predict the probability distribution of the dependent variables (the target).

For example, a simple Bayesian linear regression

.. image:: https://raw.githubusercontent.com/brendanhasz/probflow/master/docs/img/regression_equation.svg?sanitize=true
   :width: 30 %
   :align: center

can be built by creating a ProbFlow Model:

.. code-block:: python

    import probflow as pf
    import tensorflow as tf

    class LinearRegression(pf.ContinuousModel):

        def __init__(self):
            self.weight = pf.Parameter(name='weight')
            self.bias = pf.Parameter(name='bias')
            self.std = pf.ScaleParameter(name='sigma')

        def __call__(self, x):
            return pf.Normal(x*self.weight()+self.bias(), self.std())

    model = LinearRegression()

Then, the model can be fit using stochastic variational inference, in *one line*:

.. code-block:: python

    # x and y are Numpy arrays or pandas DataFrame/Series
    model.fit(x, y)

You can generate predictions for new data:

.. code-block:: pycon

    # x_test is a Numpy array or pandas DataFrame
    >>> model.predict(x_test)
    [0.983]

Compute *probabilistic* predictions for new data, with 95% confidence intervals:

.. code-block:: python

    model.pred_dist_plot(x_test, ci=0.95)

.. image:: https://raw.githubusercontent.com/brendanhasz/probflow/master/docs/img/pred_dist.svg?sanitize=true
   :width: 90 %
   :align: center

Evaluate your model's performance using metrics:

.. code-block:: pycon

    >>> model.metric('mse', x_test, y_test)
    0.217

Inspect the posterior distributions of your fit model's parameters, with 95% confidence intervals:

.. code-block:: python

    model.posterior_plot(ci=0.95)

.. image:: https://raw.githubusercontent.com/brendanhasz/probflow/master/docs/img/posteriors.svg?sanitize=true
   :width: 90 %
   :align: center

Investigate how well your model is capturing uncertainty by examining how accurate its predictive intervals are:

.. code-block:: pycon

    >>> model.pred_dist_coverage(ci=0.95)
    0.903

and diagnose *where* your model is having problems capturing uncertainty:

.. code-block:: python

    model.coverage_by(ci=0.95)

.. image:: https://raw.githubusercontent.com/brendanhasz/probflow/master/docs/img/coverage.svg?sanitize=true
   :width: 90 %
   :align: center

ProbFlow also provides more complex modules, such as those required for building Bayesian neural networks.  Also, you can mix ProbFlow with TensorFlow (or PyTorch!) code.  For example, even a somewhat complex multi-layer Bayesian neural network like this:

.. image:: https://raw.githubusercontent.com/brendanhasz/probflow/master/docs/img/dual_headed_net.svg?sanitize=true
   :width: 99 %
   :align: center

Can be built and fit with ProbFlow in only a few lines:

.. code-block:: python

    class DensityNetwork(pf.ContinuousModel):

        def __init__(self, units, head_units):
            self.core = pf.DenseNetwork(units)
            self.mean = pf.DenseNetwork(head_units)
            self.std  = pf.DenseNetwork(head_units)

        def __call__(self, x):
            x = self.core(x)
            return pf.Normal(self.mean(x), tf.exp(self.std(x)))

    # Create the model
    model = DensityNetwork([x.shape[1], 256, 128], [128, 64, 32, 1])

    # Fit it!
    model.fit(x, y)


For convenience, ProbFlow also includes several `pre-built models <http://probflow.readthedocs.io/en/latest/api_applications.html>`_ for standard tasks (such as linear regressions, logistic regressions, and multi-layer dense neural networks).  For example, the above linear regression example could have been done with much less work by using ProbFlow's ready-made LinearRegression model:

.. code-block:: python

    model = pf.LinearRegression(x.shape[1])
    model.fit(x, y)

And a multi-layer Bayesian neural net can be made easily using ProbFlow's ready-made DenseRegression model:

.. code-block:: python

    model = pf.DenseRegression([x.shape[1], 128, 64, 1])
    model.fit(x, y)

Using parameters and distributions as simple building blocks, ProbFlow allows
for the painless creation of more complicated Bayesian models like `generalized
linear models <http://probflow.readthedocs.io/en/latest/example_glm.html>`_,
`deep time-to-event models
<http://probflow.readthedocs.io/en/latest/example_time_to_event.html>`_,
`neural matrix factorization
<http://probflow.readthedocs.io/en/latest/example_nmf.html>`_ models, and
`Gaussian mixture models
<http://probflow.readthedocs.io/en/latest/example_gmm.html>`_.  You can even
mix `probabilistic and non-probabilistic models
<http://probflow.readthedocs.io/en/latest/neural_linear.html>`_!  Take a look
at the `examples <http://probflow.readthedocs.io/en/latest/examples.html>`_ and
the `user guide <http://probflow.readthedocs.io/en/latest/user_guide.html>`_
for more!


Installation
------------

Before installing ProbFlow, you'll first need to install either PyTorch, or TensorFlow 2.0 and TensorFlow Probability.  See `more details here <http://probflow.readthedocs.io/en/latest/#installation>`_.

Then, you can install ProbFlow itself:

.. code-block:: bash

    pip install probflow


Support
-------

Post bug reports, feature requests, and tutorial requests in `GitHub issues <http://github.com/brendanhasz/probflow/issues>`_.


Contributing
------------

`Pull requests <http://github.com/brendanhasz/probflow/pulls>`_ are totally welcome!  Any contribution would be appreciated, from things as minor as pointing out typos to things as major as writing new applications and distributions.


Why the name, ProbFlow?
-----------------------

Because it's a package for probabilistic modeling, and it was built on TensorFlow.  ¯\\_(ツ)_/¯
