.. _example_lme:

Mixed Effects Models
====================

.. include:: macros.hrst


TODO: description... 

TODO: math

.. math::

    \mathbf{y} \sim \text{Normal}(\mathbf{X}\beta + \mathbf{Z}\mu, \sigma)

TODO: where :math:`\mathbf{X}` are the fixed variables, :math:`\mathbf{Z}` are the random variables, :math:`\beta` is the vector of fixed-effects coefficients, and :math:`\mu` is the vector of random effects coefficients.

Manually with One-hot Encoding
------------------------------

TODO: diagram

.. code-block:: python

    class LinearMixedEffectsModel(pf.Model):

        def __init__(self, Fd, Rd):
            self.Fd = Fd
            self.beta = pf.Parameter([Fd, 1])
            self.mu = pf.Parameter([Rd, 1])
            self.sigma = pf.ScaleParameter()

        def __call__(self, x):
            beta = self.beta()
            mu = self.mu()
            X = x[:, :self.Fd]
            Z = x[:, self.Fd:]
            return pf.Normal(X @ beta + Z @ mu, self.sigma())


Using the Embedding Module
--------------------------

TODO: explain how you can instead use a 1d embedding module to model random effects (and then you don't have to one-hot encode the input data).  

TODO: below code assumes you have one random effect (IDs in last col of X)

.. code-block:: python

    class LinearMixedEffectsModel(pf.Model):

        def __init__(self, Fd, Nr):
            self.beta = pf.Parameter([Fd, 1])
            self.emb = pf.Embedding(Nr, 1)
            self.sigma = pf.ScaleParameter()

        def __call__(self, x):
            preds = x[:, :-1] @ self.beta() + self.emb(x[:, -1])
            return pf.Normal(preds, self.sigma())
