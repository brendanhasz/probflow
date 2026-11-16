import numpy as np
import pytest

from probflow.distributions import Wishart
from probflow.utils import ops
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
    log_probs = [
        to_numpy(distribution.log_prob(value))
        for distribution in distributions
    ]
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


def test_wishart_kl_divergence():
    """Test a closed-form result, asymmetry, and the zero case."""
    p = Wishart(
        to_default_dtype(6.0),
        scale_tril=to_default_dtype([[2.0, 0.0], [1.0, 1.0]]),
    )
    q = Wishart(
        to_default_dtype(4.0),
        scale_tril=to_default_dtype([[1.0, 0.0], [0.0, 1.0]]),
    )
    expected = 97 / 6 - 6 * np.log(2) - np.log(3) - 2 * np.euler_gamma
    result = to_numpy(ops.kl_divergence(p, q))
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        to_numpy(ops.kl_divergence(p(), q())), expected, rtol=1e-5
    )
    assert not np.isclose(result, to_numpy(ops.kl_divergence(q, p)))
    np.testing.assert_allclose(to_numpy(ops.kl_divergence(p, p)), 0, atol=1e-5)


def test_wishart_kl_broadcasting():
    """The result has the broadcast batch shape of the two distributions."""
    scales_p = np.array(
        [[[[1.0, 0.0], [0.3, 1.0]]], [[[2.0, 0.0], [0.1, 1.0]]]]
    )
    scales_q = np.array([[[1.0, 0.0], [0.0, x]] for x in (1.0, 1.5, 2.0)])
    p_df = [[4.0], [6.0]]
    q_df = [5.0, 7.0, 9.0]
    p = Wishart(to_default_dtype(p_df), scale_tril=to_default_dtype(scales_p))
    q = Wishart(to_default_dtype(q_df), scale_tril=to_default_dtype(scales_q))
    actual = to_numpy(ops.kl_divergence(p, q))
    expected = [
        [
            to_numpy(
                ops.kl_divergence(
                    Wishart(
                        to_default_dtype(df_p[0]),
                        scale_tril=to_default_dtype(scales_p[i, 0]),
                    ),
                    Wishart(
                        to_default_dtype(df_q),
                        scale_tril=to_default_dtype(scales_q[j]),
                    ),
                )
            )
            for j, df_q in enumerate(q_df)
        ]
        for i, df_p in enumerate(p_df)
    ]
    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_wishart_kl_mismatched_event_shapes():
    p = Wishart(to_default_dtype(5.0), scale_tril=to_default_dtype(np.eye(2)))
    q = Wishart(to_default_dtype(5.0), scale_tril=to_default_dtype(np.eye(3)))
    with pytest.raises(ValueError):
        ops.kl_divergence(p, q)
