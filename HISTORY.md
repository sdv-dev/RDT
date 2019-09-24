# History

## 0.1.3 - 2019-09-24

### New Features

* Add attributes NullTransformer and col_meta. - Issue [#30](https://github.com/HDI-Project/RDT/issues/30) by @ManuelAlvarezC

### General Improvements

* Integrate with CodeCov - Issue [#89](https://github.com/HDI-Project/RDT/issues/89) by @csala
* Remake Sphinx Documentation - Issue [#96](https://github.com/HDI-Project/RDT/issues/96) by @JDTheRipperPC
* Improve README - Issue [#92](https://github.com/HDI-Project/RDT/issues/92) by @JDTheRipperPC
* Document RELEASE workflow - Issue [#93](https://github.com/HDI-Project/RDT/issues/93) by @JDTheRipperPC
* Add support to Python 3.7 - Issue [#38](https://github.com/HDI-Project/RDT/issues/38) by @ManuelAlvarezC
* Create way to pass HyperTransformer table dict - Issue [#45](https://github.com/HDI-Project/RDT/issues/45) by @ManuelAlvarezC

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
