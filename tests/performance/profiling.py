import threading
import timeit
import tracemalloc


class MemoryProfilerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group=group, target=target, name=name, args=args,
                                  kwargs=kwargs)
        self._peak_memory = 0

    def run(self):
        tracemalloc.start()
        self._target(*self._args, **self._kwargs)
        self._peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._peak_memory


def _profile_time(method, dataset, iterations=100):
    total_time = 0
    for _ in range(iterations):
        start_time = timeit.default_timer()
        method(dataset)
        total_time += timeit.default_timer() - start_time

    return total_time / iterations


def _profile_memory(method, dataset):
    profiler_thread = MemoryProfilerThread(target=method, args=(dataset,))
    profiler_thread.start()
    peak_memory = profiler_thread.join()
    return peak_memory


def profile_transformers(transformers, datasets):
    """
    Function to profile a transformer.

    This function will get the total time and peak memory
    for the ``fit``, ``transform`` and ``reverse_transform``
    methods of the provided transformer against the provided
    dataset.

    Args:
        transformer (list):
            List of transformer class names.
        datasets (list):
            List of datasets to run transformers against.
    """
    pass


if __name__ == '__main__':
    pass
