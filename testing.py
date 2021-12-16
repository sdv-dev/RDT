import time

start_time = time.time()
pytest "tests/performance/test_performance.py::test_performance[NumericalTransformer-RandomIntegerGenerator]"
print(time.time() - start_time)

