"""Functions for saving and loading ProbFlow  objects"""

import cloudpickle

__all__ = [
    'dumps',
    'loads',
    'dump',
    'load',
]


def dumps(obj):
    """Serialize a probflow object to bytes"""
    return cloudpickle.dumps(obj)


def loads(s):
    """Deserialize a probflow object from bytes"""
    return cloudpickle.loads(s)


def dump(obj, filename):
    """Serialize a probflow object to file"""
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)


def load(filename):
    """Deserialize a probflow object from file"""
    with open(filename, "rb") as f:
        return cloudpickle.load(f)
