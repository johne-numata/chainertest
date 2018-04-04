# 2017/09 Tsuguo Mogami 
# ThreadPoolExecutor requires Python 3
# based on: https://github.com/chainer/chainer/blob/v2.0.2/chainer/iterators/multiprocess_iterator.py and serial_iterator.py

import numpy
import chainer
from concurrent.futures import ThreadPoolExecutor, wait

class MultithreadingIterator(chainer.iterators.SerialIterator):

    """Dataset iterator that loads examples in parallel.
    This is an implementation of :class:`~chainer.dataset.Iterator` that loads
    examples with worker threads. It uses the standard ThreadPoolExecutor
    class to parallelize the loading. 
    Note that this iterator effectively prefetches the examples for the next
    batch asynchronously after the current batch is returned.
    Args:
        dataset (~chainer.dataset.Dataset): Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.
        n_processes (int): Number of worker threads.
        n_prefetch (int): Not used. Spared for drop-in compatibility with multiprocess iteraotr.
        shared_mem (int): Not used. Spared for drop-in compatibility with multiprocess iteraotr.
    """
    
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True,
                 n_processes=2, n_prefetch=1, shared_mem=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._prefetch_order = None  # used at the end of each epoch

        self.n_processes = n_processes # or multiprocessing.cpu_count()
        self.n_prefetch = max(n_prefetch, 1)

        self._finalized = None
        self.reset()
        # このインスタンスが破棄される時に、poolが破棄される
        self.pool = ThreadPoolExecutor(self.n_processes)
        indices = self._next_indices()
        self.future = [self.pool.submit(self.dataset.get_example, index) for index in indices]

    
    def _next_indices(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = list(range(i, i_end))
        else:
            batch = list(self._order[i:i_end])

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch.extend(list(range(rest)))
                    else:
                        batch.extend(list(self._order[:rest]))
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end
        
        return batch
    
    
    def __next__(self):
        wait(self.future)
        batch = [f.result() for f in self.future]
        # prepare the next
        indices = self._next_indices()
        self.future = [self.pool.submit(self.dataset.get_example, index) for index in indices]
        return batch
    
    
    next = __next__

