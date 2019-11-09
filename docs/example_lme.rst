.. _example_lme:

Mixed Effects / Multilevel Models
=================================

.. include:: macros.hrst


TODO: description... 




Manually with One-hot Encoding
------------------------------

TODO: one way to do it is to manually encode the variables into design 
matrices...

.. math::

    \mathbf{y} \sim \text{Normal}(\mathbf{X}\mathbf{w} + \mathbf{Z}\mathbf{u}, ~ \sigma)

where 

- :math:`\mathbf{X}` is the fixed variables' design matrix,
- :math:`\mathbf{Z}` is the random variables' design matrix,
- :math:`\mathbf{w}` is the vector of fixed-effects coefficients,
- :math:`\mathbf{u}` is the vector of random effects coefficients, and
- :math:`\sigma` is the noise standard deviation.

.. tabs::

    .. group-tab:: TensorFlow
            
        .. code-block:: python3

            import probflow as pf

            class LinearMixedEffectsModel(pf.Model):

                def __init__(self, Fd, Rd):
                    self.Fd = Fd
                    self.beta = pf.Parameter([Fd, 1])
                    self.mu = pf.Parameter([Rd, 1])
                    self.sigma = pf.ScaleParameter()

                def __call__(self, x):
                    X = x[:, :self.Fd]
                    Z = x[:, self.Fd:]
                    return pf.Normal(X @ self.beta() + Z @ self.mu(), self.sigma())

    .. group-tab:: PyTorch
            
        .. code-block:: python3

            import probflow as pf
            import torch

            class LinearMixedEffectsModel(pf.Model):

                def __init__(self, Fd, Rd):
                    self.Fd = Fd
                    self.beta = pf.Parameter([Fd, 1])
                    self.mu = pf.Parameter([Rd, 1])
                    self.sigma = pf.ScaleParameter()

                def __call__(self, x):
                    x = torch.tensor(x)
                    X = x[:, :self.Fd]
                    Z = x[:, self.Fd:]
                    return pf.Normal(X @ self.beta() + Z @ self.mu(), self.sigma())

