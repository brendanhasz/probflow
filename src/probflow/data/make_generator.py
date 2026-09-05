"""Make input a DataGenerator if not already."""

from probflow.utils.typing import TensorLike

from .array_data_generator import ArrayDataGenerator
from .data_generator import DataGenerator


def make_generator(
    x: TensorLike | DataGenerator | None = None,
    y: TensorLike | None = None,
    batch_size: int | None = None,
    shuffle: bool = False,
    test: bool = False,
    num_workers: int | None = None,
) -> DataGenerator:
    """Make input a DataGenerator if not already."""
    if isinstance(x, DataGenerator):
        return x
    else:
        dg: ArrayDataGenerator = ArrayDataGenerator(
            x,
            y,
            batch_size=batch_size,
            test=test,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return dg
