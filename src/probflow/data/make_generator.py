from .array_data_generator import ArrayDataGenerator
from .data_generator import DataGenerator


def make_generator(
    x=None,
    y=None,
    batch_size=None,
    shuffle=False,
    test=False,
    num_workers=None,
):
    """Make input a DataGenerator if not already"""
    if isinstance(x, DataGenerator):
        return x
    else:
        dg = ArrayDataGenerator(
            x,
            y,
            batch_size=batch_size,
            test=test,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return dg
