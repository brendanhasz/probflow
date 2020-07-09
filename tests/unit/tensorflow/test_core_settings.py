"""Tests the probflow.core.settings module"""


import pytest

import tensorflow as tf

from probflow.core import settings


def test_backend():
    """Tests setting and getting the backend"""

    # Default should be tensorflow
    assert settings.get_backend() == "tensorflow"

    # Should be able to change to pytorch and back
    settings.set_backend("pytorch")
    assert settings.get_backend() == "pytorch"
    settings.set_backend("tensorflow")
    assert settings.get_backend() == "tensorflow"

    # But not anything else
    with pytest.raises(ValueError):
        settings.set_backend("lalala")

    # And it has to be a str
    with pytest.raises(TypeError):
        settings.set_backend(1)


def test_datatype():
    """Tests get and set_datatype"""

    assert isinstance(settings.get_datatype(), tf.DType)
    assert settings.get_datatype() == tf.float32

    settings.set_datatype(tf.float64)
    assert isinstance(settings.get_datatype(), tf.DType)
    assert settings.get_datatype() == tf.float64
    settings.set_datatype(tf.float32)

    with pytest.raises(TypeError):
        settings.set_datatype("lala")


def test_samples():
    """Tests setting and getting the number of samples"""

    # Default should be None
    assert settings.get_samples() is None

    # Should be able to change to an int > 0
    settings.set_samples(1)
    assert settings.get_samples() == 1
    settings.set_samples(10)
    assert settings.get_samples() == 10
    settings.set_samples(None)
    assert settings.get_samples() is None

    # But not anything <1
    with pytest.raises(ValueError):
        settings.set_samples(0)
    with pytest.raises(ValueError):
        settings.set_samples(-1)

    # And it has to be an int
    with pytest.raises(TypeError):
        settings.set_samples(3.14)
    with pytest.raises(TypeError):
        settings.set_samples("lalala")


def test_flipout():
    """Tests setting and getting the flipout setting"""

    # Default should be None
    assert settings.get_flipout() is False

    # Should be able to change to True or False
    settings.set_flipout(True)
    assert settings.get_flipout() is True
    settings.set_flipout(False)
    assert settings.get_flipout() is False

    # But only bool
    with pytest.raises(TypeError):
        settings.set_flipout(3.14)
    with pytest.raises(TypeError):
        settings.set_flipout(1)
    with pytest.raises(TypeError):
        settings.set_flipout("lalala")


def test_sampling():
    """Tests the Sampling context manager"""

    # Defaults before sampling
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False

    # Default should be samples=1 and flipout=False
    with settings.Sampling():
        assert settings.get_samples() == 1
        assert settings.get_flipout() is False

    # Should return to defaults after sampling
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False

    # Should be able to set samples and flipout via kwargs
    with settings.Sampling(n=100, flipout=True):
        assert settings.get_samples() == 100
        assert settings.get_flipout() is True

    # Again should return to defaults after __exit__
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
