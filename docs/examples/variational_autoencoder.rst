Variational Autoencoder
=======================

.. include:: ../macros.hrst

TODO: intro, link to colab w/ these examples

TODO: math

TODO: diagram

TODO: talk about how this model requires adding additional KL divergence term
(for the latent representation posterior vs prior).

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python

            import probflow as pf

            class VariationalAutoencoder(pf.ContinuousModel):

                def __init__(self, dims)
                    self.encoder = pf.DenseRegression(dims, heteroscedastic=True)
                    self.decoder = pf.DenseRegression(dims[::-1], heteroscedastic=True)

                def __call__(self, x):
                    z = self.encoder(x)
                    self.add_kl_loss(z, pf.Normal(0, 1))
                    return self.decoder(z.sample())

    .. group-tab:: PyTorch

        .. code-block:: python

            import probflow as pf
            import torch

            class VariationalAutoencoder(pf.ContinuousModel):

                def __init__(self, dims)
                    self.encoder = pf.DenseRegression(dims, heteroscedastic=True)
                    self.decoder = pf.DenseRegression(dims[::-1], heteroscedastic=True)

                def __call__(self, x):
                    x = torch.tensor(x)
                    z = self.encoder(x)
                    self.add_kl_loss(z, pf.Normal(0, 1))
                    return self.decoder(z.sample())


Then we can create an instance of the model, defining the dimensionality of
each layer of the network:

.. code-block:: python

    model = VariationalAutoencoder([7, 128, 64, 32, 3])

TODO: generate some data, and then fit

.. code-block:: python

    model.fit(x, x)
