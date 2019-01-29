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


# test process_xy_data

# test test_train_split