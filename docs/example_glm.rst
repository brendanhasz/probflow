.. _example_glm:

Poisson Regression (GLM)
========================

.. include:: macros.hrst


TODO: description... 
a generalized linear model w/ a Poisson observation distribution

TODO: math

TODO: diagram

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            import probflow as pf
            import tensorflow as tf

            class PoissonRegression(pf.Model):
                
                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 1])
                    self.b = pf.Parameter([1, 1])
                
                def __call__(self, x):
                    return pf.Poisson(tf.exp(x @ self.w() + self.b()))

    .. group-tab:: PyTorch

        .. code-block:: python3

            import probflow as pf
            import torch

            class PoissonRegression(pf.Model):
                
                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 1])
                    self.b = pf.Parameter([1, 1])
                
                def __call__(self, x):
                    x = torch.tensor(x)
                    return pf.Poisson(torch.exp(x @ self.w() + self.b()))
