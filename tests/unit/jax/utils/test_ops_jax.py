import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp_jax

import probflow as pf
from probflow.utils import ops
from probflow.utils.jax_variable import JaxVariable


def close(value, expected, tol=1e-3):
    """Check a scalar or array against an expected value."""
    assert np.allclose(np.asarray(value), np.asarray(expected), atol=tol)


def test_kl_divergence():
    """Tests KL divergence for JAX distributions."""
    dist = tfp_jax.distributions.Normal(0.0, 1.0)
    close(ops.kl_divergence(dist, dist), 0.0)
    first = tfp_jax.distributions.Normal(0.0, 1.0)
    second = tfp_jax.distributions.Normal(1.0, 1.0)
    third = tfp_jax.distributions.Normal(2.0, 1.0)
    assert float(ops.kl_divergence(first, second)) > 0
    assert float(ops.kl_divergence(first, second)) < float(
        ops.kl_divergence(first, third)
    )
    dist = pf.Normal(0.0, 1.0)
    close(ops.kl_divergence(dist, dist), 0.0)


def test_shape_helpers():
    """Tests squeeze, expand_dims, and shape."""
    value = jnp.ones((3, 2, 1))
    assert ops.squeeze(value).shape == (3, 2)
    assert ops.expand_dims(jnp.ones(3), 1).shape == (3, 1)
    assert ops.expand_dims(jnp.ones(3), 0).shape == (1, 3)
    assert ops.expand_dims(value, None).shape == value.shape
    assert ops.shape(jnp.ones((5, 4, 3))) == [5, 4, 3]


def test_creation_and_random_helpers():
    """Tests tensor creation and random helpers."""
    for function, expected in [(ops.ones, 1.0), (ops.zeros, 0.0)]:
        value = function([5, 4, 3])
        assert value.shape == (5, 4, 3)
        close(value, expected)
    assert ops.full([2, 3], 4.0).shape == (2, 3)
    assert ops.eye(4).shape == (4, 4)
    assert ops.randn([5, 4]).shape == (5, 4)
    rademacher = np.asarray(ops.rand_rademacher([5, 4]))
    assert np.all(np.isin(rademacher, [-1, 1]))


def test_reductions():
    """Tests sum, product, mean, and standard deviation."""
    value = jnp.ones((5, 4, 3))
    assert ops.sum(value).shape == (5, 4)
    close(ops.sum(value), 3.0)
    close(ops.sum(value, axis=None), 60.0)
    assert ops.prod(value, axis=1).shape == (5, 3)
    close(ops.prod(jnp.array([1.1, 2.0, 3.3])), 7.26)
    close(ops.mean(jnp.array([0.9, 1.9, 2.1, 3.1])), 2.0)
    close(ops.std(jnp.array([1.0, 2.0, 3.0])), np.std([1.0, 2.0, 3.0]))


def test_elementwise_helpers():
    """Tests elementwise mathematical and activation helpers."""
    values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    close(ops.round(jnp.array([-0.9, 0.00001, 1.0, 3.14])), [-1, 0, 1, 3])
    close(ops.abs(values), [2, 1, 0, 1, 2])
    close(ops.square(values), [4, 1, 0, 1, 4])
    close(ops.sqrt(jnp.array([0.0, 1.0, 4.0])), [0, 1, 2])
    close(ops.exp(values), np.exp(np.asarray(values)))
    close(ops.relu(values), [0, 0, 0, 1, 2])
    close(ops.softplus(values), np.log1p(np.exp(np.asarray(values))))
    close(ops.sigmoid(values), 1 / (1 + np.exp(-np.asarray(values))))


def test_indexing_and_concatenation():
    """Tests gather and concatenation."""
    values = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    gathered = ops.gather(values, jnp.array([0, 1, 2, 1, 0]))
    assert gathered.shape == (5, 2)
    close(gathered, [[1, 2], [3, 4], [5, 6], [3, 4], [1, 2]])
    gathered = ops.gather(values, jnp.array([1, 0, 1, 0]), axis=1)
    assert gathered.shape == (3, 4)
    close(gathered, [[2, 1, 2, 1], [4, 3, 4, 3], [6, 5, 6, 5]])
    first = jnp.ones((2, 3, 5))
    second = jnp.zeros((2, 3, 5))
    assert ops.cat([first, second], axis=0).shape == (4, 3, 5)
    assert ops.cat([first, second], axis=1).shape == (2, 6, 5)
    assert ops.cat([first, second], axis=2).shape == (2, 3, 10)


def test_probability_transforms_and_variables():
    """Tests probability transforms and trainable variable creation."""
    values = jnp.zeros((2, 3, 5))
    transformed = ops.additive_logistic_transform(values)
    assert transformed.shape == (2, 3, 6)
    close(ops.sum(transformed, axis=-1), 1.0)
    inserted = ops.insert_col_of(values, 1.0)
    assert inserted.shape == (2, 3, 6)
    close(inserted[:, :, 0], 1.0)
    variable = ops.new_variable(jnp.ones(3))
    assert isinstance(variable, JaxVariable)
    assert variable.shape == (3,)
    close(variable, [1, 1, 1])


def test_matrix_helpers():
    """Tests transpose, reshape, copy, and Cholesky transforms."""
    value = jnp.arange(6.0).reshape((2, 3))
    assert ops.transpose(value).shape == (3, 2)
    assert ops.reshape(value, [3, 2]).shape == (3, 2)
    copied = ops.copy_tensor(value)
    close(copied, value)
    covariance = ops.log_cholesky_transform(jnp.array([0.0, 0.0, 0.0]))
    assert covariance.shape == (2, 2)
    close(covariance, jnp.eye(2))
