"""Functions for saving and loading ProbFlow  objects"""

import cloudpickle

__all__ = [
    'dumps',
    'loads',
    'dump',
    'load',
]


def dumps(obj):
    """Serialize a probflow object to bytes.

    Note
    ----
    This removes the compiled ``_train_fn`` attribute of a |Model| which is 
    either a |TensorFlow| or |PyTorch| compiled function to perform a single
    training step.  Cloudpickle can't serialize it, and after de-serializing
    will just JIT re-compile if needed.
    """
    if hasattr(obj, "_train_fn"):
        delattr(obj, "_train_fn")
    return cloudpickle.dumps(obj)


def loads(s):
    """Deserialize a probflow object from bytes"""
    return cloudpickle.loads(s)


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
