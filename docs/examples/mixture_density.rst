Mixture Density Network
=======================

.. include:: ../macros.hrst


TODO: note that last element of head_dims should be the desired number of
mixture components

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            import probflow as pf

            class MixtureDensityNetwork(pf.Model):

                def __init__(self, dims, head_dims):
                    self.core = pf.DenseNetwork(dims+[head_dims[0]])
                    self.heads = [pf.DenseNetwork(head_dims) for _ in range(3)]

                def __call__(self, x):
                    x = self.core(x)
                    preds = [h(x) for h in self.heads]
                    return pf.Mixture(pf.Normal(preds[0], preds[1]), preds[2])

    .. group-tab:: PyTorch

        .. code-block:: python3

            import probflow as pf
            import torch

            class MixtureDensityNetwork(pf.Model):

                def __init__(self, dims, head_dims):
                    self.core = pf.DenseNetwork(dims+[head_dims[0]])
                    self.heads = [pf.DenseNetwork(head_dims) for _ in range(3)]

                def __call__(self, x):
                    x = torch.tensor(x)
                    x = self.core(x)
                    preds = [h(x) for h in self.heads]
                    return pf.Mixture(pf.Normal(preds[0], preds[1]), preds[2])


TODO: cite Christopher M. Bishop, Mixture density networks, 1994
