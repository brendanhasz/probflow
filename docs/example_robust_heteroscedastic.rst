.. _example_robust_heteroscedastic:

Robust Heteroscedastic Regression
=================================

.. include:: macros.hrst


TODO: estimates both mean and uncertainty separately, and uses a cauchy dist for the "robust" observation dist 

TODO: math and diagram

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            import probflow as pf
            import tensorflow as tf

            class RobustHeteroscedasticRegression(pf.ContinuousModel):
                
                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 2])
                    self.b = pf.Parameter(2)
                    
                def __call__(self, x):
                    p = x @ self.w() + self.b()
                    means = p[:, 0]
                    stds = tf.exp(p[:, 1])
                    return pf.Cauchy(means, stds)

    .. group-tab:: PyTorch

        .. code-block:: python3

            import probflow as pf
            import torch

            class RobustHeteroscedasticRegression(pf.ContinuousModel):
                
                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 2])
                    self.b = pf.Parameter(2)
                    
                def __call__(self, x):
                    x = torch.tensor(x)
                    p = x @ self.w() + self.b()
                    means = p[:, 0]
                    stds = torch.exp(p[:, 1])
                    return pf.Cauchy(means, stds)


TODO: you could also use a t-dist + estimate the dof

.. tabs::

    .. group-tab:: TensorFlow
    
        .. code-block:: python3

            class RobustHeteroscedasticRegression(pf.ContinuousModel):
                
                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 2])
                    self.b = pf.Parameter(2)
                    self.df = pf.ScaleParameter()
                    
                def __call__(self, x):
                    p = x @ self.w() + self.b()
                    means = p[:, 0]
                    stds = tf.exp(p[:, 1])
                    return pf.StudentT(self.df(), means, stds)

    .. group-tab:: PyTorch
    
        .. code-block:: python3

            class RobustHeteroscedasticRegression(pf.ContinuousModel):
                
                def __init__(self, dims):
                    self.w = pf.Parameter([dims, 2])
                    self.b = pf.Parameter(2)
                    self.df = pf.ScaleParameter()
                    
                def __call__(self, x):
                    x = torch.tensor(x)
                    p = x @ self.w() + self.b()
                    means = p[:, 0]
                    stds = torch.exp(p[:, 1])
                    return pf.StudentT(self.df(), means, stds)