"""Tests the probflow.utils.settings module when backend = TensorFlow."""

import pytest
import tensorflow as tf

from probflow.utils import settings


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