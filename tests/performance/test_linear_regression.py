import time
from itertools import product

import numpy as np
import pandas as pd

import probflow as pf

EPOCHS = 100
BATCH_SIZE = 1024
ns = [1024, 8192, 65536]
ds = [1, 2, 10, 100]
backends = ["pytorch", "tensorflow"]
eagers = [True, False]
# TODO: dtype (float32 or float64)
# TODO: n_mc_samples (when that's implemented)


def get_data(N, D, dtype="float32"):
    x = np.random.randn(N, D).astype(dtype)
    w = np.random.randn(D, 1).astype(dtype)
    y = x @ w + 0.1 * np.random.randn(D, 1).astype(dtype)
    return x, y


def test_linear_regression_times():

    times = []

    for N, D, backend, eager in product(ns, ds, backends, eagers):
        pf.set_backend(backend)
        model = pf.LinearRegression(D)
        x, y = get_data(N, D)
        model.fit(
            x, y, epochs=2, eager=eager
        )  # don't include compilation time
        t0 = time.time()
        model.fit(x, y, epochs=EPOCHS, eager=eager)
        t1 = time.time()
        times.append(
            {
                "N": N,
                "D": D,
                "backend": backend,
                "eager": eager,
                "seconds": t1 - t0,
            }
        )

    df = pd.DataFrame.from_records(times)
    print(df)


if __name__ == "__main__":
    test_linear_regression_times()
