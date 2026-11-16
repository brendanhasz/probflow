import numpy as np
import pytest

from probflow.distributions import Wishart
from probflow.utils.casting import to_default_dtype, to_numpy
from probflow.utils.validation import (
    is_backend_distribution,
    is_backend_tensor,
)


def test_Wishart():
    """Tests the Wishart distribution."""
    df = to_default_dtype(5.0)
    covariance = np.array([[2.0, 0.5], [0.5, 1.5]])
    precision = np.linalg.inv(covariance)
    scale_tril = np.linalg.cholesky(covariance)

    distributions = [
        Wishart(df, covariance_matrix=to_default_dtype(covariance)),
        Wishart(df, precision_matrix=to_default_dtype(precision)),
        Wishart(df, scale_tril=to_default_dtype(scale_tril)),
    ]

    for distribution in distributions:
        assert is_backend_distribution(distribution())

    value = to_default_dtype([[3.0, 0.2], [0.2, 2.0]])
    log_probs = [to_numpy(distribution.log_prob(value)) for distribution in distributions]
    assert np.allclose(log_probs[0], log_probs[1])
    assert np.allclose(log_probs[0], log_probs[2])

    samples = distributions[0].sample()
    assert is_backend_tensor(samples)
    assert samples.shape == (2, 2)
    samples = distributions[0].sample(10)
    assert is_backend_tensor(samples)
    assert samples.shape == (10, 2, 2)

    # needs at least one of covariance_matrix, precision_matrix, or scale_tril
    with pytest.raises(ValueError):
        Wishart(df)
    # df must be a number
    with pytest.raises(TypeError):
        Wishart("df", covariance_matrix=to_default_dtype(covariance))
    # covariance_matrix, precision_matrix, and scale_tril must be valid matrices
    with pytest.raises(TypeError):
        Wishart(df, covariance_matrix="covariance")
    with pytest.raises(TypeError):
        Wishart(df, precision_matrix="precision")
    with pytest.raises(TypeError):
        Wishart(df, scale_tril="scale_tril")
