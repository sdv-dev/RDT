# History

## 0.1.2

* Add a numerical transformer for positive numbers.
* Add option to anonymize data on categorical transformer.
* Move the `col_meta` argument from method-level to class-level.
* Move the logic for missing values from the transformers into the `HyperTransformer`.
* Removed unreacheble lines in `NullTransformer`.
* `Numbertransfomer` to set default value to 0 when the column is null.
* Add a CLA for collaborators.
* Refactor performance-wise the transformers.

## 0.1.1

* Improve handling of NaN in NumberTransformer and CatTransformer.
* Add unittests for HyperTransformer.
* Remove unused methods `get_types` and `impute_table` from HyperTransformer.
* Make NumberTransformer enforce dtype int on integer data.
* Make DTTransformer check data format before transforming.
* Add minimal API Reference.
* Merge `rdt.utils` into `HyperTransformer` class. 

## 0.1.0

* First release on PyPI.
