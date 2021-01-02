Models
======

.. include:: ../macros.hrst

TODO: what a model is (takes a Tensor (or nothing!) as input, and returns a probability distribution(s))

TODO: creating your own model (via a class which inherits pf.Model,
implement __init__ and __call__, and sometimes it may be required to implement
log_likelihood)

TODO: types of models and their ABCs (continuous, discrete, categorical)

.. _specifying-the-observation-distribution:

Specifying the observation distribution
---------------------------------------

The return value of ``Model.__call__`` should be a |Distribution| object, which
corresponds to the observation distribution.  The model itself predicts the
shape of this observation distribution.  For example, for a linear regression
the weights and the bias predict the mean of the observation distribution, and
the standard deviation parameter predicts the standard deviation:

.. code-block:: python3

    class LinearRegression(pf.ContinuousModel):

        def __init__(self):
            self.w = pf.Parameter()
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            mean = x * self.w() + self.b()
            std = self.s()
            return pf.Normal(mean, std)  # returns predicted observation distribution


.. _model_bayesian_updating

Bayesian updating
-----------------

Bayesian updating consists of updating a model's parameters' priors to match
their posterior distributions, after having observed some data.  This is, after
all, the main point of Bayesian inference! Because ProbFlow uses variational
inference, both parameters' posteriors and priors have an analytical form (i.e.
they're both known probability distributions with known parameters - not a set
of MCMC samples or something), and so we can literally just set parameters'
prior distribution variables to be equal to the current posterior distribution
variables!

To perform a Bayesian update of all parameters in a model, use the
:meth:`.Model.bayesian_update` method.

.. code-block:: python3

    model = # your ProbFlow model
    model.bayesian_updating()

This can be used for incremental model updates when we get more data, but don't
want to have to retrain the model from scratch on all the historical data.  For
example,

.. code-block:: python3

    # x, y = training data
    model.fit(x, y)

    # Perform the Bayesian updating!
    model.bayesian_updating()

    # x_new, y_new = new data
    model.fit(x_new, y_new)
    model.bayesian_update()

    # x_new2, y_new2 = even more new data
    model.fit(x_new2, y_new2)
    model.bayesian_update()

    # etc


Manually computing the log likelihood
-------------------------------------

The default loss function uses a log likelihood which is simply the log
probability of the observed data according to the observation distribution (the
distribution which was returned by the ``__call__`` method of the model,
:ref:`see above <specifying-the-observation-distribution>`).

However, you can override this default by re-defining the
:meth:`.Model.log_likelihood` method if you need more flexibility in how you're
computing the log probability.

For example, if you want to limit the log probability of each datapoint to -10
(essentially, perform gradient clipping):

.. code-block:: python3

    import probflow as pf
    import tensorflow as tf

    class LinearRegressionWithClipping(pf.ContinuousModel):

        def __init__(self):
            self.w = pf.Parameter()
            self.b = pf.Parameter()
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            mean = x * self.w() + self.b()
            std = self.s()
            return pf.Normal(mean, std)

        def log_likelihood(self, x, y):
            log_likelihoods = tf.math.maximum(self(x).log_prob(y), -10)
            return tf.reduce_sum(log_likelihoods)

Also see the :doc:`/examples/time_to_event` example for an example using a
custom ``log_likelihood`` method to handle censored data.
