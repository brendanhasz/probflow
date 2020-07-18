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
