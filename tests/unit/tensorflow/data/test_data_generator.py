import numpy as np

import probflow as pf
from probflow.data import *


def test_DataGenerator_workers():
    """Tests probflow.data.DataGenerator w/ multiple worker processes"""

    # Data
    x = np.random.randn(100, 3).astype("float32")
    w = np.random.randn(3, 1).astype("float32")
    y = x @ w

    # Fit a model with 1 worker
    model = pf.LinearRegression(3)
    model.fit(x, y, batch_size=10, epochs=10, num_workers=1)

    # Fit a model with 4 workers
    model = pf.LinearRegression(3)
    model.fit(x, y, batch_size=10, epochs=10, num_workers=4)
