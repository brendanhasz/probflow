"""Tests probflow.parameters modules"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.parameters import Parameter, ScaleParameter
from probflow.distributions import Normal, StudentT


def test_parameter_build():
    """Tests probflow.parameters.Parameter._build"""
    p1 = Parameter(name='test_parameter_build')
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(p1._built_prior, tfd.Normal)
    assert isinstance(p1._built_posterior, tfd.Normal)
    assert isinstance(p1._built_obj_raw, tf.Tensor)
    assert isinstance(p1.built_obj, tf.Tensor)
    assert isinstance(p1._mean_obj_raw, tf.Tensor)
    assert isinstance(p1.mean_obj, tf.Tensor)
    assert isinstance(p1._log_loss, tf.Tensor)
    assert isinstance(p1._mean_log_loss, tf.Tensor)
    assert isinstance(p1._kl_loss, tf.Tensor)


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
    assert o1 == 1.0
    assert o2 > 0.0
    assert o3 < 0.0
    assert o4 > 0.0
    assert o4 < 1.0


def test_parameter_built_obj_none_estimator():
    """Tests probflow.parameters.Parameter.built_obj w/ estimator=None"""
    p1 = Parameter(name='test_parameter_built_obj_none_estimator',
                   shape=[3,4], estimator=None)
    p1.build(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    assert o1.ndim == 3
    assert o1.shape[0] == 2
    assert o1.shape[1] == 3
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 2
    assert o2.shape[1] == 3
    assert o2.shape[2] == 4
    assert np.all(o1==o2) #no transform should have happened


def test_parameter_built_obj_flipout_estimator():
    """Tests probflow.parameters.Parameter.built_obj w/ estimator=flipout"""
    p1 = Parameter(name='test_parameter_built_obj_flipout_estimator',
                   shape=[3,4], estimator='flipout')
    p1.build(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    assert o1.ndim == 3
    assert o1.shape[0] == 2
    assert o1.shape[1] == 3
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 2
    assert o2.shape[1] == 3
    assert o2.shape[2] == 4
    assert np.all(o1==o2) #no transform should have happened


def test_parameter_mean_obj():
    """Tests probflow.parameters.Parameter.mean_obj and _mean_obj_raw"""
    p1 = Parameter(name='test_parameter_mean_obj',
                   shape=[3,4], estimator=None)
    p1.build(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2] = sess.run([p1.mean_obj, 
                             p1._mean_obj_raw])
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    assert o1.ndim == 3
    assert o1.shape[0] == 1 #"batch_size" = 1 for mean obj
    assert o1.shape[1] == 3
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 1
    assert o2.shape[1] == 3
    assert o2.shape[2] == 4
    assert np.all( o1 == o2 ) #no transform should have happened


def test_parameter_losses():
    """Tests probflow.parameters.Parameter._log_loss,_mean_log_loss,_kl_loss"""
    p1 = Parameter(name='test_parameter_losses',
                   shape=[3,4], estimator=None)
    p1.build(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2, o3] = sess.run([p1._log_loss, 
                                 p1._mean_log_loss,
                                 p1._kl_loss])
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.float32)
    assert isinstance(o3, np.float32)
    assert o1.ndim == 1
    assert o1.shape[0] == 2 #batch_size = 2


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


def test_scale_parameter_built_obj_none_estimator():
    """Tests probflow.parameters.Parameter.built_obj w/ estimator=None"""
    p1 = ScaleParameter(name='test_scale_parameter_built_obj_none_estimator',
                        shape=[3,4])
    p1.build(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    assert o1.ndim == 3
    assert o1.shape[0] == 2
    assert o1.shape[1] == 3
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 2
    assert o2.shape[1] == 3
    assert o2.shape[2] == 4
    assert not np.any(o1==o2) #transform should have happened
    # so none should be equal (unless == exactly 1.0 or 0.0)


def test_scale_parameter_mean_obj():
    """Tests probflow.parameters.Parameter.mean_obj and _mean_obj_raw"""
    p1 = ScaleParameter(name='test_scale_parameter_mean_obj',
                        shape=[3,4])
    p1.build(tf.placeholder(tf.float32, [1]), [2])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        [o1, o2] = sess.run([p1.mean_obj, 
                             p1._mean_obj_raw])
    assert isinstance(o1, np.ndarray)
    assert isinstance(o2, np.ndarray)
    assert o1.ndim == 3
    assert o1.shape[0] == 1 #"batch_size" = 1 for mean obj
    assert o1.shape[1] == 3
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 1
    assert o2.shape[1] == 3
    assert o2.shape[2] == 4
    assert not np.any(o1==o2) #transform should have happened
    # so none should be equal (unless == exactly 1.0 or 0.0)


def test_parameter_ops_overloading():
    """Tests the basic arithmetic ops are overloaded for parameters too"""

    # Param to work with
    p1 = Parameter()

    # Add
    o1 = p1 + 1
    o1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(o1.built_obj, tf.Tensor)

    # Sub
    o1 = p1 - 1
    o1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(o1.built_obj, tf.Tensor)

    # Mult
    o1 = p1 * 1
    o1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(o1.built_obj, tf.Tensor)

    # Div
    o1 = p1 + 1
    o1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(o1.built_obj, tf.Tensor)

    # Neg
    o1 = -p1
    o1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(o1.built_obj, tf.Tensor)

    # Abs
    o1 = abs(p1)
    o1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(o1.built_obj, tf.Tensor)


def test_parameter_prior_via_lshift_op():
    """Tests that param << dist sets param's prior to dist (via __lshift__)"""

    p1 = Parameter()
    assert isinstance(p1.prior, Normal)

    p1 << StudentT(1, 2, 3)

    # Should have changed the prior!
    assert not isinstance(p1.prior, Normal)
    assert isinstance(p1.prior, StudentT)
    assert p1.prior.args['df'] == 1
    assert p1.prior.args['loc'] == 2
    assert p1.prior.args['scale'] == 3

    # And should still be able to be built
    p1 << Normal(1, 3)
    p1.build(tf.placeholder(tf.float32, [1]), [1])
    assert isinstance(p1._built_prior, tfd.Normal)
    with tf.Session() as sess:
        assert sess.run(p1._built_prior.mean()) == 1.0
        assert sess.run(p1._built_prior.stddev()) == 3.0
