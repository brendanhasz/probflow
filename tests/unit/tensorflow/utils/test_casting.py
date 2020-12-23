import numpy as np
import pandas as pd
import tensorflow as tf

from probflow.utils import casting


def test_to_numpy():

    df = pd.DataFrame(np.random.randn(5, 2))
    o = casting.to_numpy(df)
    assert isinstance(o, np.ndarray)
    assert o.ndim == 2
    assert o.shape[0] == 5
    assert o.shape[1] == 2

    x = tf.random.normal([5, 2])
    o = casting.to_numpy(x)
    assert isinstance(o, np.ndarray)
    assert o.ndim == 2
    assert o.shape[0] == 5
    assert o.shape[1] == 2


def test_to_tensor():

    df = pd.DataFrame(np.random.randn(5, 2))
    o = casting.to_tensor(df)
    assert o.ndim == 2
    assert o.shape[0] == 5
    assert o.shape[1] == 2

    df = pd.Series(np.random.randn(5))
    o = casting.to_tensor(df)
    assert o.ndim == 2
    assert o.shape[0] == 5
    assert o.shape[1] == 1

    x = tf.random.normal([5, 2])
    o = casting.to_tensor(x)
    assert o.ndim == 2
    assert o.shape[0] == 5
    assert o.shape[1] == 2
