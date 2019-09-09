"""Tests probflow.utils.metrics module when backend = tensorflow"""



import pytest

import numpy as np
import tensorflow as tf

from probflow.distributions import *
from probflow.utils import metrics



def is_close(a, b, tol=1e-5):
    return np.abs(a-b) < tol



def test_accuracy():
    """Tests probflow.utils.metrics.accuracy"""

    # Predictive dist
    probs = tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32)
    pred_dist = Bernoulli(probs=probs)
    y_true = np.array([1, 0, 1, 1, 0, 1]).astype('float32')

    # Compare metric
    assert is_close(metrics.accuracy(y_true, pred_dist.mean()), 2.0/3.0)

    

def test_mean_squared_error():
    """Tests probflow.utils.metrics.mean_squared_error"""

    # Predictive dist
    preds = tf.constant([0, 1, 2, 0, 0, 0], dtype=tf.float32)
    pred_dist = Normal(preds, 1)
    y_true = np.array([0, 0, 0, 0, 1, 2]).astype('float32')

    # Compare metric
    assert is_close(metrics.mean_squared_error(y_true, pred_dist.mean()), 10.0/6.0)



def test_sum_squared_error():
    """Tests probflow.utils.metrics.sum_squared_error"""

    # Predictive dist
    preds = tf.constant([0, 1, 2, 0, 0, 0], dtype=tf.float32)
    pred_dist = Normal(preds, 1)
    y_true = np.array([0, 0, 0, 0, 1, 2]).astype('float32')

    # Compare metric
    assert is_close(metrics.sum_squared_error(y_true, pred_dist.mean()), 10.0)



def test_mean_absolute_error():
    """Tests probflow.utils.metrics.mean_absolute_error"""

    # Predictive dist
    preds = tf.constant([0, 1, 2, 0, 0, 0], dtype=tf.float32)
    pred_dist = Normal(preds, 1)
    y_true = np.array([0, 0, 0, 0, 1, 2]).astype('float32')

    # Compare metric
    assert is_close(metrics.mean_absolute_error(y_true, pred_dist.mean()), 1.0)



def test_r_squared():
    """Tests probflow.utils.metrics.r_squared"""

    # Predictive dist
    preds = tf.constant([0, 1, 2, 2, 2], dtype=tf.float32)
    pred_dist = Normal(preds, 1)
    y_true = np.array([0, 1, 2, 3, 4]).astype('float32')

    # Compare metric
    assert is_close(metrics.r_squared(y_true, pred_dist.mean()), 0.5)



def test_true_positive_rate():
    """Tests probflow.utils.metrics.true_positive_rate"""

    # Predictive dist
    probs = tf.constant([1, 1, 1, 1, 1, 0], dtype=tf.float32)
    pred_dist = Bernoulli(probs=probs)
    y_true = np.array([1, 0, 1, 1, 0, 1]).astype('float32')

    # Compare metric
    assert is_close(metrics.true_positive_rate(y_true, pred_dist.mean()), 0.75)



def test_true_negative_rate():
    """Tests probflow.utils.metrics.true_negative_rate"""

    # Predictive dist
    probs = tf.constant([1, 1, 1, 1, 1, 0], dtype=tf.float32)
    pred_dist = Bernoulli(probs=probs)
    y_true = np.array([1, 0, 1, 1, 0, 0]).astype('float32')

    # Compare metric
    assert is_close(metrics.true_negative_rate(y_true, pred_dist.mean()), 1.0/3.0)

    

def test_precision():
    """Tests probflow.utils.metrics.precision"""

    # Predictive dist
    probs = tf.constant([1, 1, 1, 1, 1, 0], dtype=tf.float32)
    pred_dist = Bernoulli(probs=probs)
    y_true = np.array([1, 0, 1, 1, 0, 0]).astype('float32')

    # Compare metric
    assert is_close(metrics.precision(y_true, pred_dist.mean()), 3.0/5.0)



def test_f1_score():
    """Tests probflow.utils.metrics.f1_score"""

    # Predictive dist
    probs = tf.constant([0, 1, 1, 1, 1, 0], dtype=tf.float32)
    pred_dist = Bernoulli(probs=probs)
    y_true = np.array([1, 0, 1, 1, 0, 0]).astype('float32')

    # Compare metric
    ppv = 2/4
    tpr = 2/3
    f1 = 2*(ppv*tpr)/(ppv+tpr)
    assert is_close(metrics.f1_score(y_true, pred_dist.mean()), f1)



def test_get_metric_fn():
    """Tests probflow.utils.metrics.get_metric_fn"""

    metric_fn = metrics.get_metric_fn('f1')
    
    # Predictive dist
    probs = tf.constant([0, 1, 1, 1, 1, 0], dtype=tf.float32)
    pred_dist = Bernoulli(probs=probs)
    y_true = np.array([1, 0, 1, 1, 0, 0]).astype('float32')

    # Compare metric
    ppv = 2/4
    tpr = 2/3
    f1 = 2*(ppv*tpr)/(ppv+tpr)
    assert is_close(metric_fn(y_true, pred_dist.mean()), f1)
    
    # Should be able to pass a callable
    metric_fn = metrics.get_metric_fn(lambda x, y: 3)
    assert metric_fn(y_true, pred_dist.mean()) == 3
    
    # Should raise a type error if passed anything else
    with pytest.raises(TypeError):
        metrics.get_metric_fn(3)
    with pytest.raises(TypeError):
        metrics.get_metric_fn([1, 2, 3])
    with pytest.raises(TypeError):
        metrics.get_metric_fn({'apples': 1, 'oranges': 2})

    # And value error if invalid string
    with pytest.raises(ValueError):
        metrics.get_metric_fn('asdf')

