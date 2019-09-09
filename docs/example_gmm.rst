.. _example_gmm:

Gaussian Mixture Model
======================

.. include:: macros.hrst


TODO: description... 

TODO: math

TODO: diagram

.. code-block:: python

    class GaussianMixtureModel(pf.Model):

        def __init__(self, d, k):
            self.m = pf.Parameter([d, k])
            self.s = pf.ScaleParameter([d, k])
            self.w = pf.CategoricalParameter([k])

        def __call__(self, x):
            means = self.m()
            stds = self.s()
            dists = [pf.Normal(means[..., i], stds[..., i])
                     for i in range(self.k)]
            return pf.Mixture(dists, self.w())
