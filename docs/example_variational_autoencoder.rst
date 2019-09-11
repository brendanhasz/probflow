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
            self.mean_encoder = pf.DenseNetwork(dims)
            self.std_encoder = pf.DenseNetwork(dims)
            self.mean_decoder = pf.DenseNetwork(dims.reverse())
            self.std_decoder = pf.DenseNetwork(dims.reverse())

        def __call__(self, x):

            # Encode
            z_mu = self.mean_encoder(x)
            z_std = tf.exp(self.std_encoder(x))
            z = z_mu+z_std*tf.random.normal(z_mu.shape)

            # Add loss due to latent prior
            self.add_kl_loss(
                tfd.kl_divergence(
                    tfd.Normal(z_mu, z_std),
                    tfd.Normal(0, 1)))

            # Decode
            mu = self.mean_decoder(z)
            std = tf.exp(self.std_decoder(z))
            return pf.Normal(mu, std)

Then we can create an instance of the model, defining the dimensionality of
each layer of the network:

.. code-block:: python

    model = VariationalAutoencoder([7, 128, 64, 32, 3])

TODO: generate some data, and then fit

.. code-block:: python

    model.fit(x, y)
