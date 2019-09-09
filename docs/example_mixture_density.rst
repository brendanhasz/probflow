.. _example_mixture_density:

Mixture Density Network
=======================

.. include:: macros.hrst


TODO: Need to implement a Mixture Distribution first!

.. code-block:: python

    class MixtureDensityNetwork(pf.Model):
        
        def __init__(self, dims, head_dims, k):
            self.k = k
            self.core = pf.DenseNetwork(dims+[head_dims[-1]])
            self.heads = [[pf.DenseNetwork(head_dims) for _ in range(k)] 
                          for _ in range(3)]
     
        def __call__(self, x):
            x = self.core(x)
            preds = [[h(x) for h in head] for head in self.heads]
            dists = [pf.Normal(preds[0][k], preds[1][k]) for k in range(self.k)]
            weights = tf.concat(preds[2], -1)
            return pf.Mixture(dists, weights)
