"""Functions for saving and loading ProbFlow  objects"""

import base64
from pathlib import Path

import cloudpickle

__all__ = [
    "dump",
    "dumps",
    "load",
    "loads",
]


def dumps(obj: object) -> str:
    """Serialize a probflow object to a json-safe string.

    Note
    ----
    This removes the compiled ``_train_fn`` attribute of a |Model| which is
    either a |TensorFlow| or |PyTorch| compiled function to perform a single
    training step.  Cloudpickle can't serialize it, and after de-serializing
    will just JIT re-compile if needed.
    """
    if "_train_fn" in obj.__dict__:
        delattr(obj, "_train_fn")
    return base64.b64encode(cloudpickle.dumps(obj)).decode("utf8")


def loads(s: str) -> object:
    """Deserialize a probflow object from string"""
    return cloudpickle.loads(base64.b64decode(s.encode("utf8")))


def dump(obj: object, filename: str | Path) -> None:
    """Serialize a probflow object to file

    Note
    ----
    This removes the compiled ``_train_fn`` attribute of a |Model| which is
    either a |TensorFlow| or |PyTorch| compiled function to perform a single
    training step.  Cloudpickle can't serialize it, and after de-serializing
    will just JIT re-compile if needed.
    """
    if "_train_fn" in obj.__dict__:
        delattr(obj, "_train_fn")
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)


def load(filename: str | Path) -> object:
    """Deserialize a probflow object from file"""
    with open(filename, "rb") as f:
        return cloudpickle.load(f)
