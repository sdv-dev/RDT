# RDT Performance Tests

This subpackage contains the performance tests for RDT.

## Generating Performance Tests

We can automatically generate new performance tests with the
`tests.performance.test_performance.make_test_case_configs` function.
This function creates a test config JSON file for the specified
(transformer, dataset, size) combinations.

It will automatically run each of the transformer methods for the indicated
number of iterations, and calculate the expected memory and runtime
thresholds from the observed resource usage.

Here is an example of how to generate test cases:

```
In [1]: from tests.performance.test_performance import make_test_case_configs

In [2]: import rdt

In [3]: from tests.performance.datasets import numerical

In [4]: dataset_generators = [
   ...: numerical.RandomIntegerGenerator,
   ...: numerical.RandomIntegerNaNsGenerator,
   ...: numerical.NormalGenerator,
   ...: numerical.NormalNaNsGenerator,
   ...: ]

In [5]: transformers = [
   ...: rdt.transformers.NumericalTransformer,
   ...: (rdt.transformers.NumericalTransformer, {'rounding': 'auto', 'min_value': 'auto', 'max_value': 'auto'}, 'auto')
   ...: ]

In [6]: sizes = [(1000, 1000), (10_000, 10_000)]

In [7]: make_test_case_configs(transformers, dataset_generators, sizes, iterations=10)
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_RandomIntegerGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_RandomIntegerGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_RandomIntegerNaNsGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_RandomIntegerNaNsGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_NormalGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_NormalGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_NormalNaNsGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/default_NormalNaNsGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_RandomIntegerGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_RandomIntegerGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_RandomIntegerNaNsGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_RandomIntegerNaNsGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_NormalGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_NormalGenerator_10000_10000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_NormalNaNsGenerator_1000_1000.json
Test case created: tests/performance/test_cases/numerical/NumericalTransformer/auto_NormalNaNsGenerator_10000_10000.json
```
