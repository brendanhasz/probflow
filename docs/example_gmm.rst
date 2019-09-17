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
            self.d = d
            self.k = k
            self.m = pf.Parameter([d, k])
            self.s = pf.ScaleParameter([d, k])
            self.w = pf.DirichletParameter(k)

        def __call__(self):
            dists = pf.Normal(self.m(), self.s())
            weights = tf.broadcast_to(self.w(), [self.d, self.k])
            return pf.Mixture(weights, dists)
