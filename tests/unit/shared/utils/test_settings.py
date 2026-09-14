"""Tests the probflow.utils.settings module."""

import uuid

import pytest
import tensorflow as tf

from probflow.utils import settings


def test_backend(monkeypatch):
    """Tests setting and getting the backend."""

    def get_mock_find_spec(
        tensorflow_installed,
        tensorflow_probability_installed,
        pytorch_installed,
    ):
        def mock_find_spec(package_name):
            installed = {
                "tensorflow": tensorflow_installed,
                "tensorflow_probability": tensorflow_probability_installed,
                "torch": pytorch_installed,
            }
            return object() if installed.get(package_name, False) else None

        return mock_find_spec

    # Freshly initialized settings should not have a backend chosen yet.
    settings.__SETTINGS__._BACKEND = None
    assert settings.__SETTINGS__._BACKEND is None

    # If tensorflow is installed, should default to TF backend
    monkeypatch.setattr(
        settings.importlib.util,
        "find_spec",
        get_mock_find_spec(
            tensorflow_installed=True,
            tensorflow_probability_installed=True,
            pytorch_installed=False,
        ),
    )
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW

    # If pytorch is installed and TF is not, should default to pytorch backend
    monkeypatch.setattr(
        settings.importlib.util,
        "find_spec",
        get_mock_find_spec(
            tensorflow_installed=False,
            tensorflow_probability_installed=False,
            pytorch_installed=True,
        ),
    )
    settings.__SETTINGS__._BACKEND = None
    assert settings.get_backend() is settings.ProbflowBackend.PYTORCH

    # If no backend is installed, should show warning but continue w/ TF
    monkeypatch.setattr(
        settings.importlib.util,
        "find_spec",
        get_mock_find_spec(
            tensorflow_installed=False,
            tensorflow_probability_installed=False,
            pytorch_installed=False,
        ),
    )
    settings.__SETTINGS__._BACKEND = None
    with pytest.warns(UserWarning, match="No backend is installed"):
        assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW

    # Should be able to change to pytorch and back using enums.
    settings.set_backend(settings.ProbflowBackend.PYTORCH)
    assert settings.get_backend() is settings.ProbflowBackend.PYTORCH
    settings.set_backend(settings.ProbflowBackend.TENSORFLOW)
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW

    # Should also work when passed the correct strings.
    settings.set_backend("pytorch")
    assert settings.get_backend() is settings.ProbflowBackend.PYTORCH
    settings.set_backend("tensorflow")
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW

    # But not any invalid string
    with pytest.raises(ValueError):
        settings.set_backend("lalala")

    # And it has to be a str or ProbflowBackend
    with pytest.raises(TypeError):
        settings.set_backend(1)


def test_datatype():
    """Tests get and set_datatype."""
    assert isinstance(settings.get_datatype(), tf.DType)
    assert settings.get_datatype() == tf.float32

    settings.set_datatype(tf.float64)
    assert isinstance(settings.get_datatype(), tf.DType)
    assert settings.get_datatype() == tf.float64
    settings.set_datatype(tf.float32)

    with pytest.raises(TypeError):
        settings.set_datatype("lala")


def test_samples():
    """Tests setting and getting the number of samples."""
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
    """Tests setting and getting the flipout setting."""
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


def test_static_sampling_uuid():
    """Tests setting and getting the static sampling uuid."""
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
    """Tests the Sampling context manager."""
    # Defaults before sampling
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None

    # Default should be Not to change anything
    with settings.Sampling():
        assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
        assert settings.get_samples() is None
        assert settings.get_flipout() is False
        assert settings.get_static_sampling_uuid() is None

    # Should be able to set samples and flipout via kwargs
    with settings.Sampling(n=100, flipout=True):
        assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
        assert settings.get_samples() == 100
        assert settings.get_flipout() is True
        assert settings.get_static_sampling_uuid() is None

    # Should return to defaults after sampling
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None

    # Should be able to set static sampling uuid
    with settings.Sampling(static=True):
        assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
        assert settings.get_samples() is None
        assert settings.get_flipout() is False
        assert settings.get_static_sampling_uuid() is not None
        s1 = settings.get_static_sampling_uuid()
        s2 = settings.get_static_sampling_uuid()
        assert (
            s1 == s2
        )  # repeated calls to get_static_sampling_uuid should be identical
        assert isinstance(settings.get_static_sampling_uuid(), uuid.UUID)

    # Without sampling, there should be no static_sampling_uuid
    with settings.Sampling(static=False):
        assert settings.get_static_sampling_uuid() is None

    # Should return to defaults after sampling
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None

    # Should be able to nest sampling context managers
    with settings.Sampling(static=True):
        assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
        assert settings.get_samples() is None
        assert settings.get_flipout() is False
        assert settings.get_static_sampling_uuid() is not None
        assert isinstance(settings.get_static_sampling_uuid(), uuid.UUID)
        with settings.Sampling(n=100, flipout=True):
            assert (
                settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
            )
            assert settings.get_samples() == 100
            assert settings.get_flipout() is True
            assert settings.get_static_sampling_uuid() is not None
            assert isinstance(settings.get_static_sampling_uuid(), uuid.UUID)

    # Should return to defaults after sampling
    assert settings.get_backend() is settings.ProbflowBackend.TENSORFLOW
    assert settings.get_samples() is None
    assert settings.get_flipout() is False
    assert settings.get_static_sampling_uuid() is None
