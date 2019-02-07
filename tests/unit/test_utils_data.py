"""Tests functions in probflow.utils.data"""

import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *
from probflow.utils import data


class DummyClass(object):
    pass


def test_utils_process_data():
    """Tests probflow.utils.data.process_data"""

    x1 = np.random.randn(20)
    x = np.random.randn(10, 3)

    # Should raise Runtime error if model is not fit
    with pytest.raises(RuntimeError):
        xo = data.process_data(Normal(0, 1))

    # Should raise TypeError error if passed data is not a ndarray
    with pytest.raises(TypeError):
        xo = data.process_data(Normal(0, 1), 3)

    # Should return model._train['x'] if passed None
    model = DummyClass()
    model._train = dict()
    model._train['x'] = x1
    xo = data.process_data(model)
    assert isinstance(xo, np.ndarray)

    # Should transform to (N,1) if 1d
    assert xo.ndim == 2
    assert xo.shape[0] == 20
    assert xo.shape[1] == 1

    # Should return x if passed x and data is none
    xo = data.process_data(model, x)
    assert isinstance(xo, np.ndarray)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 3


def test_utils_process_data_pandas():
    """Tests probflow.utils.data.process_data works w/ pandas DataFrames"""

    x = np.random.randn(10, 3)
    x[:,0] = 1.0
    x[:,2] = 2.0
    df = pd.DataFrame(x, columns=['a', 'b', 'c'])

    model = DummyClass()

    # Should raise TypeError error if passed data is not a int, str, or list
    with pytest.raises(TypeError):
        xo = data.process_data(model, 3.555, df)

    # Should raise TypeError error if passed data is not a int, str, or list
    with pytest.raises(TypeError):
        xo = data.process_data(model, [0, 3.555], df)

    # Should return col of pandas dataframe if x is string
    xo = data.process_data(model, 'a', df)
    assert isinstance(xo, np.ndarray)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 1
    assert all(xo==1.0)

    # Should return 2 cols of pandas dataframe if x is list of string
    xo = data.process_data(model, ['a', 'c'], df)
    assert isinstance(xo, np.ndarray)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 2
    assert all(xo[:,0]==1.0)
    assert all(xo[:,1]==2.0)

    # Should return col of pandas dataframe if x is int
    xo = data.process_data(model, 0, df)
    assert isinstance(xo, np.ndarray)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 1
    assert all(xo==1.0)

    # Should return 2 cols of pandas dataframe if x is list of int
    xo = data.process_data(model, [0, 2], df)
    assert isinstance(xo, np.ndarray)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 2
    assert all(xo[:,0]==1.0)
    assert all(xo[:,1]==2.0)


def test_utils_process_xy_data():
    """Tests probflow.utils.data.process_xy_data"""

    # Should transform to (N,1) if 1d
    x = np.random.randn(10)
    y = np.random.randn(10)
    model = DummyClass()
    xo, yo = data.process_xy_data(model, x, y)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 1
    assert yo.ndim == 2
    assert yo.shape[0] == 10
    assert yo.shape[1] == 1

    # Should rase a TypeError if one is None
    with pytest.raises(TypeError):
        xo, yo = data.process_xy_data(model, x, None)
    with pytest.raises(TypeError):
        xo, yo = data.process_xy_data(model, None, y)

    # Should raise a ValueError if not same size
    y = np.random.randn(5)
    with pytest.raises(ValueError):
        xo, yo = data.process_xy_data(model, x, y)

    # Keep dims if 2d
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    model = DummyClass()
    xo, yo = data.process_xy_data(model, x, y)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 3
    assert yo.ndim == 2
    assert yo.shape[0] == 10
    assert yo.shape[1] == 1


def test_utils_process_xy_data_pandas():
    """Tests probflow.utils.data.process_xy_data w/ data=pandas df"""

    # Should transform to (N,1) if 1d
    x = np.random.randn(10, 4)
    x[:,0] = 1.0
    x[:,2] = 2.0
    x[:,3] = 5.0
    df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
    model = DummyClass()
    xo, yo = data.process_xy_data(model, ['a', 'b', 'c'], 'd', df)
    assert xo.ndim == 2
    assert xo.shape[0] == 10
    assert xo.shape[1] == 3
    assert yo.ndim == 2
    assert yo.shape[0] == 10
    assert yo.shape[1] == 1


def test_test_train_split():
    """Tests probflow.utils.data.test_train_split"""

    # Data
    x = np.random.randn(10, 4)
    y = np.random.randn(10, 1)

    # No split
    N, xt, yt, xv, yv = data.test_train_split(x, y, 0.0, True)
    assert isinstance(N, int)
    assert N == 10
    assert isinstance(xt, np.ndarray)
    assert isinstance(yt, np.ndarray)
    assert isinstance(xv, np.ndarray)
    assert isinstance(yv, np.ndarray)
    assert xt.shape[0] == 10
    assert yt.shape[0] == 10
    assert xv.shape[0] == 10
    assert yv.shape[0] == 10
    assert xt.shape[1] == 4
    assert yt.shape[1] == 1
    assert xv.shape[1] == 4
    assert yv.shape[1] == 1
    assert np.all(xt == x)
    assert np.all(xv == x)
    assert np.all(yt == y)
    assert np.all(yv == y)

    # 70/30 split, no shuffle
    N, xt, yt, xv, yv = data.test_train_split(x, y, 0.3, False)
    assert N == 7
    assert isinstance(xt, np.ndarray)
    assert isinstance(yt, np.ndarray)
    assert isinstance(xv, np.ndarray)
    assert isinstance(yv, np.ndarray)
    assert xt.shape[0] == 7
    assert yt.shape[0] == 7
    assert xv.shape[0] == 3
    assert yv.shape[0] == 3
    assert xt.shape[1] == 4
    assert yt.shape[1] == 1
    assert xv.shape[1] == 4
    assert yv.shape[1] == 1
    assert np.all(xt == x[:7,:])
    assert np.all(xv == x[7:,:])
    assert np.all(yt == y[:7,:])
    assert np.all(yv == y[7:,:])

    # 70/30 split, yes shuffle
    N, xt, yt, xv, yv = data.test_train_split(x, y, 0.3, True)
    assert N == 7
    assert isinstance(xt, np.ndarray)
    assert isinstance(yt, np.ndarray)
    assert isinstance(xv, np.ndarray)
    assert isinstance(yv, np.ndarray)
    assert xt.shape[0] == 7
    assert yt.shape[0] == 7
    assert xv.shape[0] == 3
    assert yv.shape[0] == 3
    assert xt.shape[1] == 4
    assert yt.shape[1] == 1
    assert xv.shape[1] == 4
    assert yv.shape[1] == 1
    assert not np.all(xt == x[:7,:]) # it *could* happen, but unlikely...
    assert not np.all(xv == x[7:,:])
    assert not np.all(yt == y[:7,:])
    assert not np.all(yv == y[7:,:])


def test_initialize_shuffles():
    """Tests probflow.utils.data.initialize_shuffles"""

    # No shuffling
    shuff_ids = data.initialize_shuffles(10, 3, False)
    assert isinstance(shuff_ids, np.ndarray)
    assert shuff_ids.ndim == 2
    assert shuff_ids.shape[0] == 10
    assert shuff_ids.shape[1] == 3
    assert shuff_ids[0,0] == 0
    assert shuff_ids[1,0] == 1
    assert shuff_ids[0,1] == 0
    assert shuff_ids[1,1] == 1
    assert shuff_ids[0,2] == 0
    assert shuff_ids[1,2] == 1

    # Shuffling
    shuff_ids = data.initialize_shuffles(1000, 3, True)
    assert isinstance(shuff_ids, np.ndarray)
    assert shuff_ids.ndim == 2
    assert shuff_ids.shape[0] == 1000
    assert shuff_ids.shape[1] == 3
    assert not np.all(shuff_ids[:10,0] == np.arange(10))
    assert not np.all(shuff_ids[:10,0] == shuff_ids[:10,1])
    assert not np.all(shuff_ids[:10,1] == shuff_ids[:10,2])


def test_generate_batch():
    """Tests probflow.utils.data.generate_batch"""

    N = 10
    batch_size = 4
    x = np.reshape(np.arange(N*3), (N, 3), order='F')
    y = np.reshape(np.arange(N), (N,1))
    shuff_ids = data.initialize_shuffles(N, 3, False)
    tx, ty, tb = data.generate_batch(x, y, 0, 0, batch_size, shuff_ids)
    assert isinstance(tx, np.ndarray)
    assert isinstance(ty, np.ndarray)
    assert isinstance(tb, list)
    assert len(tb) == 1
    assert isinstance(tb[0], int)
    assert tb[0] == 4
    assert tx.ndim == 2
    assert tx.shape[0] == 4
    assert tx.shape[1] == 3
    assert ty.ndim == 2
    assert ty.shape[0] == 4
    assert ty.shape[1] == 1
    assert np.all(tx[:,0] == np.arange(4))
    assert np.all(tx[:,1] == np.arange(10, 14))
    assert np.all(tx[:,2] == np.arange(20, 24))
    assert np.all(ty[:,0] == np.arange(4))

    # Generate another batch
    tx, ty, tb = data.generate_batch(x, y, 0, 1, batch_size, shuff_ids)
    assert tb[0] == 4
    assert tx.shape[0] == 4
    assert tx.shape[1] == 3
    assert ty.shape[0] == 4
    assert ty.shape[1] == 1
    assert np.all(tx[:,0] == np.arange(4, 8))
    assert np.all(tx[:,1] == np.arange(14, 18))
    assert np.all(tx[:,2] == np.arange(24, 28))
    assert np.all(ty[:,0] == np.arange(4, 8))

    # Generate another batch
    tx, ty, tb = data.generate_batch(x, y, 0, 2, batch_size, shuff_ids)
    assert tb[0] == 2
    assert tx.shape[0] == 2
    assert tx.shape[1] == 3
    assert ty.shape[0] == 2
    assert ty.shape[1] == 1
    assert np.all(tx[:,0] == np.arange(8, 10))
    assert np.all(tx[:,1] == np.arange(18, 20))
    assert np.all(tx[:,2] == np.arange(28, 30))
    assert np.all(ty[:,0] == np.arange(8, 10))

    # Generate first batch of second epoch
    tx, ty, tb = data.generate_batch(x, y, 1, 0, batch_size, shuff_ids)
    assert tb[0] == 4
    assert tx.shape[0] == 4
    assert tx.shape[1] == 3
    assert ty.shape[0] == 4
    assert ty.shape[1] == 1
    assert np.all(tx[:,0] == np.arange(4))
    assert np.all(tx[:,1] == np.arange(10, 14))
    assert np.all(tx[:,2] == np.arange(20, 24))
    assert np.all(ty[:,0] == np.arange(4))
