"""
TODO: Data utilities, more info...

* :class:`.DataGenerator` - base class for data generators w/ multiprocessing
* :class:`.ArrayDataGenerator` - data generator for array-structured data

----------

"""


__all__ = [
    "DataGenerator",
    "ArrayDataGenerator",
    "make_generator",
]


from .data_generator import DataGenerator
from .array_data_generator import ArrayDataGenerator
from .make_generator import make_generator
