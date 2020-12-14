import multiprocessing as mp
from abc import abstractmethod

from probflow.utils.base import BaseDataGenerator


class DataGenerator(BaseDataGenerator):
    """Abstract base class for a data generator, which uses multiprocessing
    to load the data in parallel.

    TODO

    User needs to implement:

    * :meth:`~__init__`
    * :meth:`~n_samples`
    * :meth:`~batch_size`
    * :meth:`~get_batch`

    And can optionally implement:

    * :meth:`~on_epoch_start`
    * :meth:`~on_epoch_end`

    """

    def __init__(self, num_workers=None):
        self.num_workers = num_workers

    @abstractmethod
    def get_batch(self, index):
        """Generate one batch of data"""

    def __getitem__(self, index):
        """Generate one batch of data"""

        # No multiprocessing
        if self.num_workers is None:

            return self.get_batch(index)

        # Multiprocessing
        else:

            # Start the next worker
            pid = index + self.num_workers
            if pid < len(self):
                self._workers[pid].start()

            # Return data from the multiprocessing queue
            return self._queue.get()

    def __iter__(self):
        """Get an iterator over batches"""

        # Multiprocessing?
        if self.num_workers is not None:

            def get_data(index, queue):
                queue.put(self.get_batch(index))

            # Create the queue and worker processes
            self._queue = mp.Queue()
            self._workers = [
                mp.Process(target=get_data, args=(i, self._queue))
                for i in range(len(self))
            ]

            # Start the first num_workers workers
            for i in range(min(self.num_workers, len(self))):
                self._workers[i].start()

        # Keep track of what batch we're on
        self._batch = -1

        # Return iterator
        return self

    def __next__(self):
        """Get the next batch"""
        self._batch += 1
        if self._batch < len(self):
            return self[self._batch]
        else:
            raise StopIteration()
