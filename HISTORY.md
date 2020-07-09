# History

## 0.2.3 - 2020-07-09

* Implement OneHot and Label encoding as transformers - Issue [#112](https://github.com/sdv-dev/RDT/issues/112) by @csala

## 0.2.2 - 2020-06-26

### Bugs Fixed

* Escape `column_name` in hypertransformer - Issue [#110](https://github.com/sdv-dev/RDT/issues/110) by @csala

## 0.2.1 - 2020-01-17

### Bugs Fixed

* Boolean Transformer fails to revert when there are NO nulls - Issue [#103](https://github.com/sdv-dev/RDT/issues/103) by @JDTheRipperPC

## 0.2.0 - 2019-10-15

This version comes with a brand new API and internal implementation, removing the old
metadata JSON from the user provided arguments, and making each transformer work only
with `pandas.Series` of their corresponding data type.

As part of this change, several transformer names have been changed and a new BooleanTransformer
and a feature to automatically decide which transformers to use based on dtypes have been added.

Unit test coverage has also been increased to 100%.

Special thanks to @JDTheRipperPC and @csala for the big efforts put in making this
release possible.

### Issues

* Drop the usage of meta - Issue [#72](https://github.com/sdv-dev/RDT/issues/72) by @JDTheRipperPC
* Make CatTransformer.probability_map deterministic - Issue [#25](https://github.com/sdv-dev/RDT/issues/25) by @csala

## 0.1.3 - 2019-09-24

### New Features

* Add attributes NullTransformer and col_meta - Issue [#30](https://github.com/sdv-dev/RDT/issues/30) by @ManuelAlvarezC

### General Improvements

* Integrate with CodeCov - Issue [#89](https://github.com/sdv-dev/RDT/issues/89) by @csala
* Remake Sphinx Documentation - Issue [#96](https://github.com/sdv-dev/RDT/issues/96) by @JDTheRipperPC
* Improve README - Issue [#92](https://github.com/sdv-dev/RDT/issues/92) by @JDTheRipperPC
* Document RELEASE workflow - Issue [#93](https://github.com/sdv-dev/RDT/issues/93) by @JDTheRipperPC
* Add support to Python 3.7 - Issue [#38](https://github.com/sdv-dev/RDT/issues/38) by @ManuelAlvarezC
* Create way to pass HyperTransformer table dict - Issue [#45](https://github.com/sdv-dev/RDT/issues/45) by @ManuelAlvarezC

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
