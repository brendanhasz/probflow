.. _example_glm:

Poisson Regression (GLM)
========================

.. include:: macros.hrst


TODO: description... a generalized linear model w/ a Poisson observation distribution

TODO: math

TODO: diagram

.. code-block:: python

    import probflow as pf

    class PoissonRegression(pf.Model):
        
        def __init__(self, dims):
            self.w = pf.Parameter([dims, 1])
            self.b = pf.Parameter([1, 1])
        
        def __call__(self, x):
            return pf.Poisson(tf.exp(x @ self.w() + self.b()))
