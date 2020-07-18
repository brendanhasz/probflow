"""Functions for saving and loading ProbFlow  objects"""

import base64

import cloudpickle

__all__ = [
    "dumps",
    "loads",
    "dump",
    "load",
]


def dumps(obj):
    """Serialize a probflow object to a json-safe string.

    Note
    ----
    This removes the compiled ``_train_fn`` attribute of a |Model| which is
    either a |TensorFlow| or |PyTorch| compiled function to perform a single
    training step.  Cloudpickle can't serialize it, and after de-serializing
    will just JIT re-compile if needed.
    """
    if hasattr(obj, "_train_fn"):
        delattr(obj, "_train_fn")
    return base64.b64encode(cloudpickle.dumps(obj)).decode("utf8")


def loads(s):
    """Deserialize a probflow object from string"""
    return cloudpickle.loads(base64.b64decode(s.encode("utf8")))


def dump(obj, filename):
    """Serialize a probflow object to file

    Note
    ----
    This removes the compiled ``_train_fn`` attribute of a |Model| which is
    either a |TensorFlow| or |PyTorch| compiled function to perform a single
    training step.  Cloudpickle can't serialize it, and after de-serializing
    will just JIT re-compile if needed.
    """
    if hasattr(obj, "_train_fn"):
        delattr(obj, "_train_fn")
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)


def load(filename):
    """Deserialize a probflow object from file"""
    with open(filename, "rb") as f:
        return cloudpickle.load(f)
