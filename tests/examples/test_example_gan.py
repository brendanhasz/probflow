"""Tests example GAN model"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import probflow as pf

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_gmm():
    """Tests gaussian mixture model example"""

    class Generator(pf.Model):
        def __init__(self, dims):
            self.Dz = dims[0]
            self.G = pf.DenseNetwork(dims)
            self.D = None

        def __call__(self, x):
            z = tf.random.normal([x.shape[0], self.Dz])
            return self.G(z)

        def log_likelihood(self, _, x):
            labels = tf.ones([x.shape[0], 1])
            true_ll = self.D(self(x)).log_prob(labels)
            return tf.reduce_sum(true_ll)

    class Discriminator(pf.Model):
        def __init__(self, dims):
            self.G = None
            self.D = pf.DenseNetwork(dims)

        def __call__(self, x):
            return pf.Bernoulli(self.D(x))

        def log_likelihood(self, _, x):
            labels = tf.ones([x.shape[0], 1])
            true_ll = self(x).log_prob(labels)
            fake_ll = self(self.G(x)).log_prob(0 * labels)
            return tf.reduce_sum(true_ll + fake_ll)

    class TrainGenerator(pf.Callback):
        def __init__(self, G, x):
            self.G = G
            self.x = x

        def on_epoch_end(self):
            self.G.fit(self.x, epochs=1)

    # Data
    Nf = 7
    Nz = 3
    x = np.random.randn(1000, Nf)

    # Create the networks
    G = Generator([Nz, 256, 128, Nf])
    D = Discriminator([Nf, 256, 128, 1])

    # Let them know about each other <3
    G.D = lambda x: D(x)
    D.G = lambda x: G(x)

    # Create the callback which trains the generator
    train_g = TrainGenerator(G, x)

    # Fit both models by fitting the discriminator w/ the callback
    D.fit(x, epochs=2, callbacks=[train_g])
