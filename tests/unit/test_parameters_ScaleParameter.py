"""Tests probflow.parameters.ScaleParameter class"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.parameters import ScaleParameter



def test_scale_parameter_built_obj():
    """Tests probflow.parameters.ScaleParameter.built_obj"""
    p1 = ScaleParameter(name='test_scale_parameter_built_obj',
                        shape=[3,4])
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
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
    """Tests probflow.parameters.ScaleParameter.mean_obj and _mean_obj_raw"""
    p1 = ScaleParameter(name='test_scale_parameter_mean_obj',
                        shape=[3,4])
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [2])
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
