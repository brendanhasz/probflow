import numpy as np

from probflow.data import ArrayDataGenerator, make_generator


def test_make_generator():

    # Create some data
    x = np.random.randn(100, 3)
    w = np.random.randn(3, 1)
    b = np.random.randn()
    y = x @ w + b

    # Should return an ArrayDataGenerator
    dg = make_generator(x, y)
    assert isinstance(dg, ArrayDataGenerator)

    # Should just return what passed if passed an ArrayDataGenerator
    dg = ArrayDataGenerator(x, y, batch_size=5)
    dg_out = make_generator(dg)
    assert isinstance(dg_out, ArrayDataGenerator)
