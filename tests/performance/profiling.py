import threading
import timeit
import tracemalloc

import pandas as pd


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


def profile_transformer(transformer, dataset_generator, transform_size, fit_size=None):
    """
    Function to profile a transformer.

    This function will get the total time and peak memory
    for the ``fit``, ``transform`` and ``reverse_transform``
    methods of the provided transformer against the provided
    dataset.

    Args:
        transformer (Transformer):
            Transformer instance.
        dataset_generator (DatsetGenerator):
            DatasetGenerator instance.
        transform_size (int):
            Number of rows to generate for ``transform`` and ``reverse_transform``.
        fit_size (int or None):
            Number of rows to generate for ``fit``. If None, use ``transform_size``.
    """
    fit_size = fit_size or transform_size
    fit_dataset = dataset_generator.generate(fit_size)
    fit_time = _profile_time(transformer.fit, fit_dataset)
    fit_memory = _profile_memory(transformer.fit, fit_dataset)

    transform_dataset = dataset_generator.generate(transform_size)
    transform_time = _profile_time(transformer.transform, transform_dataset)
    transform_memory = _profile_memory(transformer.transform, transform_dataset)

    reverse_dataset = transformer.transform(transform_dataset)
    reverse_time = _profile_time(transformer.reverse_transform, reverse_dataset)
    reverse_memory = _profile_memory(transformer.reverse_transform, reverse_dataset)

    return pd.DataFrame({
        'Fit Time': [fit_time],
        'Fit Memory': [fit_memory],
        'Transform Time': [transform_time],
        'Transform Memory': [transform_memory],
        'Reverse Transform Time': [reverse_time],
        'Reverse Transform Memory': [reverse_memory]
    })


if __name__ == '__main__':
    pass
