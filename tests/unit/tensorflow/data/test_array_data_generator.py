import numpy as np
import pandas as pd
import pytest

from probflow.data import ArrayDataGenerator


def test_ArrayDataGenerator():
    """Tests probflow.data.ArrayDataGenerator"""

    # Should error with invalid args
    with pytest.raises(TypeError):
        dg = ArrayDataGenerator(x="lala")
    with pytest.raises(TypeError):
        dg = ArrayDataGenerator(y="lala")
    with pytest.raises(TypeError):
        dg = ArrayDataGenerator(batch_size=1.1)
    with pytest.raises(ValueError):
        dg = ArrayDataGenerator(batch_size=-1)
    with pytest.raises(TypeError):
        dg = ArrayDataGenerator(shuffle=1.1)
    with pytest.raises(TypeError):
        dg = ArrayDataGenerator(test=1.1)
    with pytest.raises(ValueError):
        dg = ArrayDataGenerator(x=np.ones(3), y=np.ones(4))

    # Create some data
    x = np.random.randn(100, 3)
    w = np.random.randn(3, 1)
    b = np.random.randn()
    y = x @ w + b

    # Create the generator
    dg = ArrayDataGenerator(x, y, batch_size=5)

    # Check properties
    assert dg.n_samples == 100
    assert dg.batch_size == 5
    assert dg.shuffle is False

    # len should return # batches per epoch
    assert len(dg) == 20

    # __getitem__ should return a batch
    x1, y1 = dg[0]
    assert isinstance(x1, np.ndarray)
    assert isinstance(y1, np.ndarray)
    assert x1.shape[0] == 5
    assert x1.shape[1] == 3
    assert y1.shape[0] == 5
    assert y1.shape[1] == 1

    # should return the same data if called twice
    x2, y2 = dg[0]
    assert np.all(x1 == x2)
    assert np.all(y1 == y2)

    # but not after shuffling
    dg.shuffle = True
    dg.on_epoch_end()
    x2, y2 = dg[0]
    assert np.sum(x1 == x2) < 10
    assert np.sum(y1 == y2) < 10

    # should be able to iterate over batches
    i = 0
    for xx, yy in dg:
        assert isinstance(xx, np.ndarray)
        assert isinstance(yy, np.ndarray)
        assert xx.shape[0] == 5
        assert xx.shape[1] == 3
        assert yy.shape[0] == 5
        assert yy.shape[1] == 1
        i += 1
    assert i == 20

    # should handle if y is None
    dg = ArrayDataGenerator(y=x, batch_size=5)
    for xx, yy in dg:
        assert xx is None
        assert isinstance(yy, np.ndarray)
        assert yy.shape[0] == 5
        assert yy.shape[1] == 3
    dg.on_epoch_end()

    # and if y is None, should treat x as y (for generative models)
    dg = ArrayDataGenerator(x, batch_size=5)
    for xx, yy in dg:
        assert xx is None
        assert isinstance(yy, np.ndarray)
        assert yy.shape[0] == 5
        assert yy.shape[1] == 3

    # should be able to make an empty generator
    dg = ArrayDataGenerator()
    for xx, yy in dg:
        assert xx is None
        assert yy is None

    # should handle pandas dataframes and series
    x = np.random.randn(100, 3)
    w = np.random.randn(3, 1)
    b = np.random.randn()
    y = x @ w + b
    x = pd.DataFrame(x)
    y = pd.Series(y[:, 0])

    dg = ArrayDataGenerator(x, y, batch_size=5)
    assert dg.n_samples == 100
    assert dg.batch_size == 5
    assert dg.shuffle is False
    assert len(dg) == 20
    x1, y1 = dg[0]
    assert isinstance(x1, pd.DataFrame)
    assert isinstance(y1, pd.Series)
    assert x1.shape[0] == 5
    assert x1.shape[1] == 3
    assert y1.shape[0] == 5
    x2, y2 = dg[0]
    assert np.all(x1.values == x2.values)
    assert np.all(y1.values == y2.values)
