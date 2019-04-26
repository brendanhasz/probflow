"""Tests probflow.parameters.CategoricalParameter class"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow.parameters import CategoricalParameter



def test_categorical_parameter_built_obj_values_int():
    """Tests probflow.parameters.CategoricalParameter.built_obj
    when initialized with values as a scalar"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=3, name='test_categorical_parameter1')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])
        [l1, l2] = sess.run([p1._built_posterior.logits,
                             p1._built_prior.logits])

    # Check samples' shape
    assert o1.ndim == 2
    assert o1.shape[0] == 100
    assert o1.shape[1] == 1
    assert o2.ndim == 2
    assert o2.shape[0] == 100
    assert o2.shape[1] == 1

    # Outputs should be in [0, 1, 2]
    assert len(np.unique(o1)) == 3
    assert 0 in np.unique(o1)
    assert 1 in np.unique(o1)
    assert 2 in np.unique(o1)

    # When values is initialized as a scalar, should be no transform
    assert all(o1 == o2)

    # Prior logits should all be the same
    assert len(np.unique(l2)) == 1

    # Shape of posterior and prior logits should be the same
    assert l1.shape == l2.shape


def test_categorical_parameter_built_obj_values_list():
    """Tests probflow.parameters.CategoricalParameter.built_obj
    when initialized with values as a list"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=[-1, 0, 1], 
                              name='test_categorical_parameter2')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])

    # Check samples' shape
    assert o1.ndim == 2
    assert o1.shape[0] == 100
    assert o1.shape[1] == 1
    assert o2.ndim == 2
    assert o2.shape[0] == 100
    assert o2.shape[1] == 1

    # Outputs should be in [-1, 0, 1]
    assert len(np.unique(o1)) == 3
    assert -1 in np.unique(o1)
    assert 0 in np.unique(o1)
    assert 1 in np.unique(o1)

    # When values is initialized as a list, should transform
    assert all(o1[o2==0] == -1)
    assert all(o1[o2==1] == 0)
    assert all(o1[o2==2] == 1)


def test_categorical_parameter_built_obj_shape_values_int():
    """Tests probflow.parameters.CategoricalParameter.built_obj
    when initialized with values as a scalar, w/ shape param"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=3, shape=[2, 4],
                              name='test_categorical_parameter3')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])

    # Check samples' shape
    assert o1.ndim == 3
    assert o1.shape[0] == 100
    assert o1.shape[1] == 2
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 100
    assert o2.shape[1] == 2
    assert o2.shape[2] == 4

    # Outputs should be in [0, 1, 2]
    assert len(np.unique(o1)) == 3
    assert 0 in np.unique(o1)
    assert 1 in np.unique(o1)
    assert 2 in np.unique(o1)

    # When values is initialized as a scalar, should be no transform
    assert all((o1 == o2).ravel())


def test_categorical_parameter_built_obj_values_list():
    """Tests probflow.parameters.CategoricalParameter.built_obj
    when initialized with values as a list"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=[-1, 0, 1], shape=[2, 4],
                              name='test_categorical_parameter4')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.built_obj, 
                             p1._built_obj_raw])

    # Check samples' shape
    assert o1.ndim == 3
    assert o1.shape[0] == 100
    assert o1.shape[1] == 2
    assert o1.shape[2] == 4
    assert o2.ndim == 3
    assert o2.shape[0] == 100
    assert o2.shape[1] == 2
    assert o2.shape[2] == 4

    # Outputs should be in [-1, 0, 1]
    assert len(np.unique(o1)) == 3
    assert -1 in np.unique(o1)
    assert 0 in np.unique(o1)
    assert 1 in np.unique(o1)

    # When values is initialized as a list, should transform
    assert all(o1[o2==0] == -1)
    assert all(o1[o2==1] == 0)
    assert all(o1[o2==2] == 1)


def test_categorical_parameter_mean_obj_values_int():
    """Tests probflow.parameters.CategoricalParameter.mean_obj
    when initialized with values as a scalar"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=3, name='test_categorical_parameter5')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.mean_obj, 
                             p1._mean_obj_raw])

    # Check mean shape
    assert o1.ndim == 2
    assert o1.shape[0] == 1 #should NOT be batch_size
    assert o1.shape[1] == 1
    assert o2.ndim == 2
    assert o2.shape[0] == 1
    assert o2.shape[1] == 1

    # Outputs should be in [0, 1, 2]
    assert o1 in [0, 1, 2]

    # When values is initialized as a scalar, should be no transform
    assert o1 == o2


def test_categorical_parameter_mean_obj_values_list():
    """Tests probflow.parameters.CategoricalParameter.mean_obj
    when initialized with values as a list"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=[11, 12, 13], 
                              name='test_categorical_parameter6')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.mean_obj, 
                             p1._mean_obj_raw])

    # Check samples' shape
    assert o1.ndim == 2
    assert o1.shape[0] == 1 #should NOT be batch_size
    assert o1.shape[1] == 1
    assert o2.ndim == 2
    assert o2.shape[0] == 1
    assert o2.shape[1] == 1

    # Outputs should be in [11, 12, 13]
    assert o1 in [11, 12, 13]

    # When values is initialized as a list, should transform
    if o1 == 11:
        assert o2 == 0
    elif o1 == 12:
        assert o2 == 1
    elif o1 == 13:
        assert o2 == 2


def test_categorical_parameter_mean_obj_shape_values_int():
    """Tests probflow.parameters.CategoricalParameter.mean_obj
    when initialized with values as a scalar, w/ shape param"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=3, shape=[20, 5],
                              name='test_categorical_parameter7')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.mean_obj, 
                             p1._mean_obj_raw])

    # Check samples' shape
    assert o1.ndim == 3
    assert o1.shape[0] == 1 #should NOT be batch_shape
    assert o1.shape[1] == 20
    assert o1.shape[2] == 5
    assert o2.ndim == 3
    assert o2.shape[0] == 1
    assert o2.shape[1] == 20
    assert o2.shape[2] == 5

    # Outputs should be in [0, 1, 2]
    assert len(np.unique(o1)) == 3
    assert 0 in np.unique(o1)
    assert 1 in np.unique(o1)
    assert 2 in np.unique(o1)

    # When values is initialized as a scalar, should be no transform
    assert all((o1 == o2).ravel())


def test_categorical_parameter_mean_obj_values_list():
    """Tests probflow.parameters.CategoricalParameter.mean_obj
    when initialized with values as a list"""

    # Create and build the parameter
    p1 = CategoricalParameter(values=[11, 12, 13], shape=[20, 5],
                              name='test_categorical_parameter8')
    p1._build_recursively(tf.placeholder(tf.float32, [1]), [100])

    # Evaluate samples
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [o1, o2] = sess.run([p1.mean_obj, 
                             p1._mean_obj_raw])

    # Check samples' shape
    assert o1.ndim == 3
    assert o1.shape[0] == 1 #should NOT be batch_shape
    assert o1.shape[1] == 20
    assert o1.shape[2] == 5
    assert o2.ndim == 3
    assert o2.shape[0] == 1
    assert o2.shape[1] == 20
    assert o2.shape[2] == 5

    # Outputs should be in [11, 12, 13]
    assert len(np.unique(o1)) == 3
    assert 11 in np.unique(o1)
    assert 12 in np.unique(o1)
    assert 13 in np.unique(o1)

    # When values is initialized as a list, should transform
    assert all(o1[o2==0] == 11)
    assert all(o1[o2==1] == 12)
    assert all(o1[o2==2] == 13)
