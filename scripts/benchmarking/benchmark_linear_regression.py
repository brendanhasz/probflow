import time
import gc
from itertools import product

import numpy as np
import pandas as pd

import probflow as pf

EPOCHS = 100
TRAIN_BATCH_SIZE = 1024
PREDICT_BATCH_SIZE = 1024
SAMPLE_SIZE = 1000
MAX_N_FOR_EAGER = 1024
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


def run_single_benchmark_linear_regression(n: int, d: int, eager: bool) -> list[dict]:
    """Run a single linear regression benchmark, returning runtime metrics."""
    # Setup
    model = pf.LinearRegression(d)
    x, y = get_data(n, d)
    data = []

    # Benchmark training time
    t0 = time.time()
    model.fit(x, y, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, eager=eager)
    t1 = time.time()
    data.append({ 
        "n_datapoints": n,
        "n_dimensions": d,
        "backend": pf.get_backend().value,
        "eager": eager,
        "runtime_seconds": t1 - t0,
        "operation": "train",
    })

    # Benchmark prediction time
    t0 = time.time()
    _ = model.predict(x, batch_size=PREDICT_BATCH_SIZE)
    t1 = time.time()
    data.append({
        "n_datapoints": n,
        "n_dimensions": d,
        "backend": pf.get_backend().value,
        "eager": eager,  # NOTE: not really applicable here, but for filtering purposes...
        "runtime_seconds": t1 - t0,
        "operation": "predict",
    })

    # Benchmark sampling time
    t0 = time.time()
    _ = model.predictive_sample(x, n=SAMPLE_SIZE, batch_size=PREDICT_BATCH_SIZE)
    t1 = time.time()
    data.append({
        "n_datapoints": n,
        "n_dimensions": d,
        "backend": pf.get_backend().value,
        "eager": eager,  # NOTE: not really applicable here, but for filtering purposes...
        "runtime_seconds": t1 - t0,
        "operation": "sample",
    })

    print(data)
    return data


def benchmark_linear_regression():
    """Test linear regression times."""
    data = []
    backend = pf.get_backend()

    # For other cases, always use compiled execution (non-eager)
    for n, d, eager in product(ns, ds, eagers):
        if not eager or n <= MAX_N_FOR_EAGER:  # do not benchmark larger datasets for eager, takes too long
            data.extend(run_single_benchmark_linear_regression(n=n, d=d, eager=eager))
            gc.collect()

    df = pd.DataFrame.from_records(data)

    # Save the results to a CSV file
    df.to_csv(f"scripts/benchmarking/benchmark_linear_regression_{backend.value}.csv", index=False)


if __name__ == "__main__":
    benchmark_linear_regression()
