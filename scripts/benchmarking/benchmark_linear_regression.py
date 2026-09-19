"""Benchmark the performance of fitting a linear regression with ProbFlow."""

import gc
import sys
import time
import tracemalloc
from itertools import product

import numpy as np
import pandas as pd

import probflow as pf

EPOCHS = 100
BATCHES_PER_EPOCH = 128
MIN_BATCH_SIZE = 1024
SAMPLE_SIZE = 1000
MAX_N_FOR_EAGER = 1024
ns = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17]
ds = [100]
eagers = [True, False]


def get_data(N, D, dtype="float32"):
    """Provide get data."""
    rng = np.random.default_rng(seed=1234)
    x = rng.standard_normal((N, D)).astype(dtype)
    w = rng.standard_normal((D, 1)).astype(dtype)
    y = x @ w + 0.1 * rng.standard_normal((N, 1)).astype(dtype)
    return x, y


def run_single_benchmark_linear_regression(
    n: int,
    d: int,
    eager: bool,
    device: str,
) -> list[dict]:
    """Run a single linear regression benchmark, returning runtime metrics."""
    # Setup
    model = pf.LinearRegression(d)
    x, y = get_data(n, d)
    batch_size = max(n // BATCHES_PER_EPOCH, MIN_BATCH_SIZE)
    data = []

    # Benchmark training time
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()
    t0 = time.time()
    model.fit(x, y, epochs=EPOCHS, batch_size=batch_size, eager=eager)
    t1 = time.time()
    _, mem_max = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    data.append(
        {
            "n_datapoints": n,
            "n_dimensions": d,
            "backend": pf.get_backend().value,
            "eager": eager,
            "runtime_seconds": t1 - t0,
            "memory_usage": mem_max - mem_start,
            "operation": "train",
            "device": device,
        }
    )

    # Benchmark prediction time
    t0 = time.time()
    _ = model.predict(x, batch_size=batch_size)
    t1 = time.time()
    data.append(
        {
            "n_datapoints": n,
            "n_dimensions": d,
            "backend": pf.get_backend().value,
            "eager": eager,  # NOTE: not really applicable here, but for filtering purposes...
            "runtime_seconds": t1 - t0,
            "operation": "predict",
            "device": device,
        }
    )

    # Benchmark sampling time
    t0 = time.time()
    _ = model.predictive_sample(x, n=SAMPLE_SIZE, batch_size=batch_size)
    t1 = time.time()
    data.append(
        {
            "n_datapoints": n,
            "n_dimensions": d,
            "backend": pf.get_backend().value,
            "eager": eager,  # NOTE: not really applicable here, but for filtering purposes...
            "runtime_seconds": t1 - t0,
            "operation": "sample",
            "device": device,
        }
    )

    print(data)
    return data


def verify_tensor_device(tensor, expected_device: str):
    """Verify that the given tensor is on the expected device."""
    if pf.get_backend() == pf.ProbflowBackend.PYTORCH:
        expected_device_type_str = (
            "cuda" if expected_device == "gpu" else "cpu"
        )
        assert tensor.device.type == expected_device_type_str, (
            f"Expected device {expected_device}, got {tensor.device}"
        )
    elif pf.get_backend() == pf.ProbflowBackend.JAX:
        from probflow.utils.jax_variable import JaxVariable

        if isinstance(tensor, JaxVariable):
            tensor = tensor.value
        expected_device_type_str = (
            "cuda" if expected_device == "gpu" else "CPU"
        )
        assert expected_device_type_str in str(tensor.device), (
            f"Expected device {expected_device}, got {tensor.device}"
        )
    else:  # tensorflow
        expected_device_type_str = "GPU" if expected_device == "gpu" else "CPU"
        assert expected_device_type_str in tensor.device, (
            f"Expected device {expected_device}, got {tensor.device}"
        )


def verify_default_device(expected_device: str):
    """Verify and print the default device for the current backend."""
    # Verify default device
    if pf.get_backend() == pf.ProbflowBackend.PYTORCH:
        import torch

        default_device = torch.get_default_device()
        expected_device_type_str = (
            "cuda" if expected_device == "gpu" else "cpu"
        )
        assert expected_device_type_str in default_device.type, (
            f"Expected device {expected_device}, got {default_device}"
        )
    elif pf.get_backend() == pf.ProbflowBackend.JAX:
        import jax

        default_device = jax.default_backend()
        expected_device_type_str = "gpu" if expected_device == "gpu" else "cpu"
        assert default_device == expected_device_type_str, (
            f"Expected device {expected_device}, got {default_device}"
        )
    else:
        import tensorflow as tf

        gpu_device_name = tf.test.gpu_device_name()

        if expected_device == "gpu":
            assert "GPU" in gpu_device_name, (
                f"Expected CUDA device, got {gpu_device_name}"
            )
        elif expected_device == "cpu":
            assert gpu_device_name == "", (
                f"Expected CPU device, got {gpu_device_name}"
            )
        else:
            raise ValueError(f"Unknown device: {expected_device}")

    # Verify a ProbFlow-created parameter uses the correct device for variables
    test_param = pf.Parameter()
    verify_tensor_device(test_param.posterior.loc, expected_device)

    # Verify ops are executed on the correct device
    import probflow.utils.ops as O

    verify_tensor_device(O.ones([3]), expected_device)


def benchmark_linear_regression(device: str):
    """Test linear regression times."""
    data = []
    backend = pf.get_backend()

    # For other cases, always use compiled execution (non-eager)
    for n, d, eager in product(ns, ds, eagers):
        if (
            not eager or n <= MAX_N_FOR_EAGER
        ):  # do not benchmark larger datasets for eager, takes too long
            data.extend(
                run_single_benchmark_linear_regression(
                    n=n, d=d, eager=eager, device=device
                )
            )
            gc.collect()

    df = pd.DataFrame.from_records(data)

    # Save the results to a CSV file
    df.to_csv(
        f"scripts/benchmarking/benchmark_linear_regression_{backend.value}_{device}.csv",
        index=False,
    )


if __name__ == "__main__":
    EXPECTED_DEVICE = sys.argv[1]
    assert EXPECTED_DEVICE in ["cpu", "gpu"], (
        "Expected device must be 'cpu' or 'gpu'"
    )
    verify_default_device(expected_device=EXPECTED_DEVICE)
    benchmark_linear_regression(device=EXPECTED_DEVICE)
