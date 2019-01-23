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
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(p1._built_prior, tfd.Normal)
    assert isinstance(p1._built_posterior, tfd.Normal)
    assert isinstance(p1.seed_stream, tfd.SeedStream)

    # TODO: will have to check built_obj, mean_obj, _built_obj_raw, _mean_obj_raw, _log_loss, _mean_log_loss, and _kl_loss


def test_parameter_ensure_is_built():
    """Tests probflow.parameters.Parameter._ensure_is_built"""
    p1 = Parameter(name='test_parameter_ensure_is_built')
    with pytest.raises(RuntimeError):
        p1._ensure_is_built()
    p1.build(tf.placeholder(tf.float32, [1]), [1])
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
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    # TODO: test it works when data is a dataset iterator obj


# TODO: _sample

# TODO: _mean

# TODO: _log_loss

# TODO: _kl_loss

def test_parameter_sample_posterior():
    """Tests probflow.parameters.Parameter.sample_posterior"""

    # Scalar parameter
    p1 = Parameter(name='test_parameter_posterior')
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    init_op = tf.global_variables_initializer()
    the_sess = tf.Session()
    the_sess.run(init_op)
    p1._session = the_sess
    samples = p1.sample_posterior(num_samples=10)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim==2
    assert samples.shape[0]==10
    assert samples.shape[1]==1
    the_sess.close()

    # Parameter with shape=(2,1)
    p1 = Parameter(name='test_parameter_posterior21', shape=[2, 1])
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    init_op = tf.global_variables_initializer()
    the_sess = tf.Session()
    the_sess.run(init_op)
    p1._session = the_sess
    samples = p1.sample_posterior(num_samples=10)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim==3
    assert samples.shape[0]==10
    assert samples.shape[1]==2
    assert samples.shape[2]==1
    the_sess.close()

    # Parameter with shape=(1,2)
    p1 = Parameter(name='test_parameter_posterior12', shape=[1, 2])
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    init_op = tf.global_variables_initializer()
    the_sess = tf.Session()
    the_sess.run(init_op)
    p1._session = the_sess
    samples = p1.sample_posterior(num_samples=10)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim==3
    assert samples.shape[0]==10
    assert samples.shape[1]==1
    assert samples.shape[2]==2
    the_sess.close()
