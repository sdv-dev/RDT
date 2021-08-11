"""Functions to profile performance of RDT Transformers."""

import timeit
import tracemalloc
from copy import deepcopy
from multiprocessing import Process, Value

import numpy as np
import pandas as pd


def _profile_time(transformer, method_name, dataset, iterations=100, copy=False):
    total_time = 0
    for _ in range(iterations):
        if copy:
            transformer_copy = deepcopy(transformer)
            method = getattr(transformer_copy, method_name)

        else:
            method = getattr(transformer, method_name)

        start_time = timeit.default_timer()
        method(dataset)
        total_time += timeit.default_timer() - start_time

    return total_time / iterations


def _set_memory_for_method(method, dataset, peak_memory):
    tracemalloc.start()
    method(dataset)
    peak_memory.value = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()


def _profile_memory(method, dataset):
    peak_memory = Value('i', 0)
    profiling_process = Process(target=_set_memory_for_method, args=(method, dataset, peak_memory))
    profiling_process.start()
    profiling_process.join()
    return peak_memory.value


def profile_transformer(transformer, dataset_generator, transform_size, fit_size=None):
    """Profile a Transformer on a dataset.

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

    Returns:
        pandas.Series:
            Series containing the time and memory taken by ``fit``, ``transform``,
            and ``reverse_transform`` for the transformer.
    """
    fit_size = fit_size or transform_size
    fit_dataset = dataset_generator.generate(fit_size)
    fit_time = _profile_time(transformer, 'fit', fit_dataset, copy=True)
    fit_memory = _profile_memory(transformer.fit, fit_dataset)
    transformer.fit(fit_dataset)

    replace = transform_size > fit_size
    transform_dataset = np.random.choice(fit_dataset, transform_size, replace=replace)
    transform_time = _profile_time(transformer, 'transform', transform_dataset)
    transform_memory = _profile_memory(transformer.transform, transform_dataset)

    reverse_dataset = transformer.transform(transform_dataset)
    reverse_time = _profile_time(transformer, 'reverse_transform', reverse_dataset)
    reverse_memory = _profile_memory(transformer.reverse_transform, reverse_dataset)

    print('Fit Time', fit_time)
    print('Fit Memory', fit_memory)
    print('Transform Time', transform_time)
    print('Transform Memory', transform_memory)
    print('Reverse Transform Time', reverse_time)
    print('Reverse Transform Memory', reverse_memory)

    return pd.Series({
        'Fit Time': fit_time,
        'Fit Memory': fit_memory,
        'Transform Time': transform_time,
        'Transform Memory': transform_memory,
        'Reverse Transform Time': reverse_time,
        'Reverse Transform Memory': reverse_memory
    })
