"""Tests probflow.parameters modules"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.parameters import Parameter


def test_parameter_build():
    """Tests probflow.parameters.Parameter._build"""
    p1 = Parameter(name='test_parameter_build')
    p1._build(None)
    assert isinstance(p1._built_prior, tfd.Normal)
    assert isinstance(p1._built_posterior, tfd.Normal)
    assert isinstance(p1.seed_stream, tfd.SeedStream)


def test_parameter_ensure_is_built():
    """Tests probflow.parameters.Parameter._ensure_is_built"""
    p1 = Parameter(name='test_parameter_ensure_is_built')
    with pytest.raises(RuntimeError):
        p1._ensure_is_built()
    p1._build(None)
    p1._ensure_is_built()


def test_parameter_bound():
    """Tests probflow.parameters.Parameter._bound"""
    t1 = Parameter._bound(None, tf.Variable([1.]), None, None)
    t2 = Parameter._bound(None, tf.Variable([-1.]), 0., None)
    t3 = Parameter._bound(None, tf.Variable([1.]), None, 0.)
    t4 = Parameter._bound(None, tf.Variable([2.]), 0., 1.)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2, o3, o4] = sess.run([t1, t2, t3, t4])
    assert o1==1.0
    assert o2>0.0
    assert o3<0.0
    assert o4>0.0
    assert o4<1.0


def test_parameter_sample_none_estimator():
    """Tests probflow.parameters.Parameter._sample w/ estimator=None"""
    p1 = Parameter(name='test_parameter_sample_none_estimator',
                   estimator=None)
    p1._build(None)
    # TODO: test it works when data is a dataset iterator obj


# TODO: _sample

# TODO: _mean

# TODO: _log_loss

# TODO: _kl_loss

def test_parameter_posterior():
    """Tests probflow.parameters.Parameter.posterior"""
    p1 = Parameter(name='test_parameter_posterior')
    p1._build(None)
    init_op = tf.global_variables_initializer()
    the_sess = tf.Session()
    the_sess.run(init_op)
    p1._session = the_sess
    samples = p1.posterior(num_samples=10)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim==2
    assert samples.shape[0]==10
    assert samples.shape[1]==1
    # TODO: posterior from parameter w/ 2d shape?