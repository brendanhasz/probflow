import time
from itertools import product

import numpy as np
import pandas as pd

import probflow as pf

EPOCHS = 100
BATCH_SIZE = 1024
ns = [1024, 8192, 65536]
ds = [1, 2, 10, 100]
backends = [
    pf.ProbflowBackend.PYTORCH,
    pf.ProbflowBackend.TENSORFLOW,
    pf.ProbflowBackend.JAX,
]
eagers = [True, False]


def get_data(N, D, dtype="float32"):
    """Provide get data."""
    x = np.random.randn(N, D).astype(dtype)
    w = np.random.randn(D, 1).astype(dtype)
    y = x @ w + 0.1 * np.random.randn(D, 1).astype(dtype)
    return x, y


def benchmark_linear_regression():
    """Test linear regression times."""
    times = []
    cached_data = {}
    for N, D, backend, eager in product(ns, ds, backends, eagers):
        pf.set_backend(backend)
        model = pf.LinearRegression(D)
        if (N, D) not in cached_data:
            cached_data[(N, D)] = get_data(N, D)
        x, y = cached_data[(N, D)]
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

    # Save the results to a CSV file
    df.to_csv("benchmark_linear_regression.csv", index=False)


if __name__ == "__main__":
    benchmark_linear_regression()
