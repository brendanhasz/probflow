import numpy as np
import pytest

from probflow.modules import Embedding
from probflow.parameters import DeterministicParameter
from probflow.utils.settings import Sampling
import probflow.utils.ops as O
from probflow.utils.validation import is_backend_tensor
from probflow.utils.casting import to_numpy


def test_Embedding():
    """Tests probflow.modules.Embedding."""
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
    x = np.random.default_rng().integers(low=0, high=9, size=(20, 1)).astype(np.int32)
    samples1 = emb(x)
    samples2 = emb(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5

    # Samples should actually be the same b/c using deterministic posterior
    with Sampling(n=1):
        samples1 = emb(x)
        samples2 = emb(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5

    # kl_loss should return sum of KL losses
    kl_loss = emb.kl_loss()
    assert is_backend_tensor(kl_loss)
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
    x = np.concatenate(
        (
            np.random.default_rng().integers(low=0, high=9, size=(20, 1)).astype(np.int32),
            np.random.default_rng().integers(low=0, high=19, size=(20, 1)).astype(np.int32),
        ),
        axis=1,
    )
    samples1 = emb(x)
    samples2 = emb(x)
    assert np.all(to_numpy(samples1) == to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 9

    # With probabilistic = True, samples should be different
    emb = Embedding(10, 5, probabilistic=True)
    x = np.random.default_rng().integers(low=0, high=9, size=(20, 1)).astype(np.int32)
    with Sampling(n=1):
        samples1 = emb(x)
        samples2 = emb(x)
    assert np.all(to_numpy(samples1) != to_numpy(samples2))
    assert samples1.ndim == 2
    assert samples1.shape[0] == 20
    assert samples1.shape[1] == 5
    assert all(
        not isinstance(e, DeterministicParameter) for e in emb.embeddings
    )
