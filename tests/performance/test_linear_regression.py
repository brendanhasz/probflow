import time
from itertools import product

import numpy as np
import pandas as pd

import probflow as pf

EPOCHS = 100
BATCH_SIZE = 1024
ns = [1024, 8192, 65536]
ds = [1, 2, 10, 100]
eagers = [True, False]


def get_data(N, D, dtype="float32"):
    """Provide get data."""
    rng = np.random.default_rng(seed=1234)
    x = rng.standard_normal((N, D)).astype(dtype)
    w = rng.standard_normal((D, 1)).astype(dtype)
    y = x @ w + 0.1 * rng.standard_normal((N, 1)).astype(dtype)
    return x, y


def benchmark_linear_regression():
    """Test linear regression times."""
    data = []
    backend = pf.get_backend()
    for N, D, eager in product(ns, ds, eagers):
        model = pf.LinearRegression(D)
        x, y = get_data(N, D)
        t0 = time.time()
        model.fit(x, y, epochs=EPOCHS, eager=eager)
        t1 = time.time()
        data.append(
            {
                "N": N,
                "D": D,
                "backend": backend.value,
                "eager": eager,
                "seconds": t1 - t0,
            }
        )

    df = pd.DataFrame.from_records(data)

    # Save the results to a CSV file
    df.to_csv(f"benchmark_linear_regression_{backend.value}.csv", index=False)


if __name__ == "__main__":
    benchmark_linear_regression()
