.. _example_ppca:

Probabilistic PCA
=================

.. include:: macros.hrst


TODO: description... 

TODO: math

TODO: diagram


.. code-block:: python3

    class PPCA(pf.Model):

        def __init__(self, d, q):
            self.W = pf.Parameter(shape=[d, q])
            self.sigma = pf.ScaleParameter()

        def __call__(self):
            W = self.W()
            cov = W @ tf.transpose(W) + self.sigma()*tf.eye(W.shape[0])
            return pf.MultivariateNormal(tf.zeros(W.shape[0]), cov)
