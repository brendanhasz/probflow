.. _example_gmm:

Gaussian Mixture Model
======================

.. include:: macros.hrst


TODO: description... 

TODO: math

TODO: diagram

.. code-block:: python3

    class GaussianMixtureModel(pf.Model):

        def __init__(self, d, k):
            self.m = pf.Parameter([d, k])
            self.s = pf.ScaleParameter([d, k])
            self.w = pf.DirichletParameter(k)

        def __call__(self):
            return pf.Mixture(pf.Normal(self.m(), self.s()), probs=self.w())
