"""Tests example probabilistic PCA"""


import pytest

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import probflow as pf



def is_close(a, b, tol=1e-3):
    return np.abs(a-b) < tol



def test_gmm():
    """Tests gaussian mixture model example"""

    class PPCA(pf.Model):

        def __init__(self, d, q):
            self.W = pf.Parameter(shape=[d, q])
            self.sigma = pf.ScaleParameter()

        def __call__(self):
            W = self.W()
            cov = W @ tf.transpose(W) + self.sigma()*tf.eye(W.shape[0])
            return pf.MultivariateNormal(tf.zeros(W.shape[0]), cov)


    model = PPCA(3, 2)

    # Create some data
    X = np.concatenate([np.random.randn(100, 3), 
                        np.random.randn(100, 3)+5,
                        np.random.randn(100, 3)+10], axis=0)
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    # Fit the model
    model.fit(X)

    # TODO: check it has fit correctly
    #import pdb; pdb.set_trace()
