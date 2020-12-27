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
