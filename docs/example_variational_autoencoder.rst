.. _example_variational_autoencoder:

Variational Autoencoder
=======================

.. include:: macros.hrst

TODO: intro, link to colab w/ these examples

TODO: math

TODO: diagram

TODO: talk about how this model requires adding additional KL divergence term
(for the latent representation posterior vs prior).

.. code-block:: python

    import probflow as pf

    class VariationalAutoencoder(pf.ContinuousModel):

        def __init__(self, dims)
            self.encoder = pf.DenseNetwork(dims[:-1] + [2*dims[-1]])
            self.decoder = pf.DenseNetwork(dims[-1:0:-1] + [2*dims[0]])

        def split(self, x)
            N = x.shape[1]/2
            return x[:, :N], tf.exp(x[:, N:])

        def __call__(self, x):

            # Encode
            z_mu, z_std = self.split(self.encoder(x))

            # Sample from latent posterior via reparameterization trick
            z = z_mu+z_std*tf.random.normal(z_mu.shape)

            # Add loss due to latent prior
            self.add_kl_loss(
                tfd.kl_divergence(
                    tfd.Normal(z_mu, z_std),
                    tfd.Normal(0, 1)))

            # Decode
            mu, std = self.split(self.decoder(z))
            return pf.Normal(mu, std)


Then we can create an instance of the model, defining the dimensionality of
each layer of the network:

.. code-block:: python

    model = VariationalAutoencoder([7, 128, 64, 32, 3])

TODO: generate some data, and then fit

.. code-block:: python

    model.fit(x, y)
