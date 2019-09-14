.. _example_correlation:

Bayesian Correlation
====================

.. include:: macros.hrst


TODO: description... 

TODO: math

TODO: diagram

TODO: note that this treats it as a generative model (ie only y no x)

TODO: test that this works after implementing MultivariateNormal....

.. code-block:: python3

    class BayesianCorrelation(pf.Model):

        def __init__(self):
            self.rho = pf.BoundedParameter()

        def __call__(self):
            rho = self.rho()
            return pf.MultivariateNormal(tf.zeros([2, 2]),
                                         tf.constant([[1, rho], [rho, 1]))
