"""Tests probflow.distributions modules"""


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.distributions import *
from probflow.layers import Add


lp_1at0 = -0.9189385332046727 # log prob of normal dist @ 0sig w/ sig=1
lp_1at1 = -1.4189385332046727 # log prob of normal dist @ 1sig w/ sig=1
lp_2at0 = -1.612085713764618 # log prob of normal dist @ 0sig w/ sig=2
lp_2at1 = -2.112085713764618 # log prob of normal dist @ 1sig w/ sig=2


def isclose(a, b, tol=1e-6):
    """Returns true if a and b are w/i tol"""
    return abs(a-b) < tol


def test_normal_float_input():
    """Tests probflow.distributions.Normal w/ float/int input"""
    d1 = Normal(2.0, 1.0)
    d1.build()
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu, 
            sig, 
            lp1, 
            lp2,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(1.0),
            d1.built_obj.log_prob(2.0),
        ])
    assert mu==2.0
    assert sig==1.0
    assert isclose(lp1, lp_1at1)
    assert isclose(lp2, lp_1at0)


def test_normal_numpy_input():
    """Tests probflow.distributions.Normal w/ numpy array input"""
    a = np.array([[0], [2]]).astype('float32')
    b = np.array([[2], [1]]).astype('float32')
    d1 = Normal(a, b)
    d1.build()
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu, 
            sig, 
            lp0, 
            lp1,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.array([[0.], [2]])),
            d1.built_obj.log_prob(np.array([[-2.], [3]])),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert isinstance(lp1, np.ndarray)
    assert mu.ndim==2
    assert mu[0][0]==0.0
    assert mu[1][0]==2.0
    assert sig[0][0]==2.0
    assert sig[1][0]==1.0
    assert isclose(lp0[0][0], lp_2at0)
    assert isclose(lp0[1][0], lp_1at0)
    assert isclose(lp1[0][0], lp_2at1)
    assert isclose(lp1[1][0], lp_1at1)


def test_normal_layer_input():
    """Tests probflow.distributions.Normal w/ layer inputs"""
    d1 = Normal(Add(1.0, 1.0), Add(0.5, 0.5))
    d1.build()
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu,
            sig,
            lp1,
            lp2,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(1.0),
            d1.built_obj.log_prob(2.0),
        ])
    assert mu==2.0
    assert sig==1.0
    assert isclose(lp1, lp_1at1)
    assert isclose(lp2, lp_1at0)


def test_normal_tensor_input():
    """Tests probflow.distributions.Normal w/ tf.Tensor input"""
    a = tf.constant([[0], [2]], dtype=tf.float32)
    b = tf.constant([[2], [1]], dtype=tf.float32)
    d1 = Normal(a, b)
    d1.build()
    assert isinstance(d1.built_obj, tfd.Normal)
    with tf.Session() as sess:
        [
            mu, 
            sig, 
            lp0, 
            lp1,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.array([[0.], [2]])),
            d1.built_obj.log_prob(np.array([[-2.], [3]])),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert isinstance(lp1, np.ndarray)
    assert mu.ndim==2
    assert mu[0][0]==0.0
    assert mu[1][0]==2.0
    assert sig[0][0]==2.0
    assert sig[1][0]==1.0
    assert isclose(lp0[0][0], lp_2at0)
    assert isclose(lp0[1][0], lp_1at0)
    assert isclose(lp1[0][0], lp_2at1)
    assert isclose(lp1[1][0], lp_1at1)


def test_normal_variable_input():
    """Tests probflow.distributions.Normal w/ tf.Variable input"""
    a = tf.Variable([[0], [2]], dtype=tf.float32)
    b = tf.Variable([[2], [1]], dtype=tf.float32)
    d1 = Normal(a, b)
    d1.build()
    assert isinstance(d1.built_obj, tfd.Normal)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [
            mu, 
            sig, 
            lp0, 
            lp1,
        ] = sess.run([
            d1.built_obj.mean(),
            d1.built_obj.stddev(),
            d1.built_obj.log_prob(np.array([[0.], [2]])),
            d1.built_obj.log_prob(np.array([[-2.], [3]])),
        ])
    assert isinstance(mu, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(lp0, np.ndarray)
    assert isinstance(lp1, np.ndarray)
    assert mu.ndim==2
    assert mu[0][0]==0.0
    assert mu[1][0]==2.0
    assert sig[0][0]==2.0
    assert sig[1][0]==1.0
    assert isclose(lp0[0][0], lp_2at0)
    assert isclose(lp0[1][0], lp_1at0)
    assert isclose(lp1[0][0], lp_2at1)
    assert isclose(lp1[1][0], lp_1at1)