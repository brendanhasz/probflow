"""Tests the probflow.utils.settings module"""

import uuid

import pytest
import tensorflow as tf

from probflow.utils import settings


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

    # Default should be False
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


def test_flipout():
    """Tests setting and getting the static sampling uuid"""

    # Default should be None
    assert settings.get_static_sampling_uuid() is None

    # Should be able to change to True or False
    the_uuid = uuid.uuid4()
    settings.set_static_sampling_uuid(the_uuid)
    assert settings.get_static_sampling_uuid() is not None
    assert settings.get_static_sampling_uuid() == the_uuid
    settings.set_static_sampling_uuid(None)
    assert settings.get_static_sampling_uuid() is None

    # But only None or uuid
    with pytest.raises(TypeError):
        settings.set_static_sampling_uuid(3.14)
    with pytest.raises(TypeError):
        settings.set_static_sampling_uuid(1)
    with pytest.raises(TypeError):
        settings.set_static_sampling_uuid("lalala")


def test_sampling():
    """Tests the Sampling context manager"""

    # Defaults before sampling
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None

    # Default should be Not to change anything
    with settings.Sampling():
        assert settings.get_backend() == "tensorflow"
        assert settings.get_samples() is None
        assert settings.get_flipout() is False
        assert settings.get_static_sampling_uuid() is None

    # Should be able to set samples and flipout via kwargs
    with settings.Sampling(n=100, flipout=True):
        assert settings.get_backend() == "tensorflow"
        assert settings.get_samples() == 100
        assert settings.get_flipout() is True
        assert settings.get_static_sampling_uuid() is None

    # Should return to defaults after sampling
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None

    # Should be able to set static sampling uuid
    with settings.Sampling(static=True):
        assert settings.get_backend() == "tensorflow"
        assert settings.get_samples() is None
        assert settings.get_flipout() is False
        assert settings.get_static_sampling_uuid() is not None
        assert isinstance(settings.get_static_sampling_uuid(), uuid.UUID)

    # Should return to defaults after sampling
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None

    # Should be able to nest sampling context managers
    with settings.Sampling(static=True):
        assert settings.get_backend() == "tensorflow"
        assert settings.get_samples() is None
        assert settings.get_flipout() is False
        assert settings.get_static_sampling_uuid() is not None
        assert isinstance(settings.get_static_sampling_uuid(), uuid.UUID)
        with settings.Sampling(n=100, flipout=True):
            assert settings.get_backend() == "tensorflow"
            assert settings.get_samples() == 100
            assert settings.get_flipout() is True
            assert settings.get_static_sampling_uuid() is not None
            assert isinstance(settings.get_static_sampling_uuid(), uuid.UUID)

    # Should return to defaults after sampling
    assert settings.get_backend() == "tensorflow"
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None
