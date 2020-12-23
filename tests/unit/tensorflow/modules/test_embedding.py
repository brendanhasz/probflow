import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow.modules import Embedding
from probflow.parameters import DeterministicParameter
from probflow.utils.settings import Sampling

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_Embedding():
    """Tests probflow.modules.Embedding"""

    # Should error w/ int < 1
    with pytest.raises(ValueError):
        emb = Embedding(0, 1)
    with pytest.raises(ValueError):
        emb = Embedding(5, -1)

    # Should error w/ k and d of different lengths
    with pytest.raises(ValueError):
        emb = Embedding([2, 3], [2, 3, 4])

    # Create the module
    emb = Embedding(10, 5)

    # Check parameters
    assert len(emb.parameters) == 1
    assert emb.parameters[0].name == "Embedding_0"
    assert emb.parameters[0].shape == [10, 5]

    # Embeddings should be DeterministicParameters by default
    assert all(isinstance(e, DeterministicParameter) for e in emb.embeddings)

    # Test MAP outputs are the same
    x = tf.random.uniform([20, 1], minval=0, maxval=9, dtype=tf.dtypes.int32)
    samples1 = emb(x)
    samples2 = emb(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5

    # Samples should actually be the same b/c using deterministic posterior
    with Sampling():
        samples1 = emb(x)
        samples2 = emb(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5

    # kl_loss should return sum of KL losses
    kl_loss = emb.kl_loss()
    assert isinstance(kl_loss, tf.Tensor)
    assert kl_loss.ndim == 0

    # Should be able to embed multiple columns by passing list of k and d
    emb = Embedding([10, 20], [5, 4])

    # Check parameters
    assert len(emb.parameters) == 2
    assert emb.parameters[0].name == "Embedding_0"
    assert emb.parameters[0].shape == [10, 5]
    assert emb.parameters[1].name == "Embedding_1"
    assert emb.parameters[1].shape == [20, 4]

    # Test MAP outputs are the same
    x1 = tf.random.uniform([20, 1], minval=0, maxval=9, dtype=tf.dtypes.int32)
    x2 = tf.random.uniform([20, 1], minval=0, maxval=19, dtype=tf.dtypes.int32)
    x = tf.concat([x1, x2], axis=1)
    samples1 = emb(x)
    samples2 = emb(x)
    assert np.all(samples1.numpy() == samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 9

    # With probabilistic = True, samples should be different
    emb = Embedding(10, 5, probabilistic=True)
    x = tf.random.uniform([20, 1], minval=0, maxval=9, dtype=tf.dtypes.int32)
    with Sampling():
        samples1 = emb(x)
        samples2 = emb(x)
    assert np.all(samples1.numpy() != samples2.numpy())
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5
    assert all(
        not isinstance(e, DeterministicParameter) for e in emb.embeddings
    )
