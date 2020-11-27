# History

## 0.2.9 - 2020-11-27

This release fixes a bug that prevented the `CategoricalTransformer` from working properly
when being passed data that contained numerical data only, without any strings, but also
contained `None` or `NaN` values.

### Issues closed

* KeyError: nan - CategoricalTransformer fails on numerical + nan data only - Issue [#142](https://github.com/sdv-dev/RDT/issues/142) by @csala

## 0.2.8 - 2020-11-20

This release fixes a few minor bugs, including some which prevented RDT from fully working
on Windows systems.

Thanks to this fixes, as well as a new testing infrastructure that has been set up, from now
on RDT is officially supported on Windows systems, as well as on the Linux and macOS systems
which were previously supported.

### Issues closed

* TypeError: unsupported operand type(s) for: 'NoneType' and 'int' - Issue [#132](https://github.com/sdv-dev/RDT/issues/132) by @csala
* Example does not work on Windows - Issue [#114](https://github.com/sdv-dev/RDT/issues/114) by @csala
* OneHotEncodingTransformer producing all zeros - Issue [#135](https://github.com/sdv-dev/RDT/issues/135) by @fealho
* OneHotEncodingTransformer support for lists and lists of lists - Issue [#137](https://github.com/sdv-dev/RDT/issues/137) by @fealho

## 0.2.7 - 2020-10-16

In this release we drop the support for the now officially dead Python 3.5
and introduce a new feature in the DatetimeTransformer which reduces the dimensionality
of the generated numerical values while also ensuring that the reverted datetimes
maintain the same level as time unit precision as the original ones.

* Drop Py35 support - Issue [#129](https://github.com/sdv-dev/RDT/issues/129) by @csala
* Add option to drop constant parts of the datetimes - Issue [#130](https://github.com/sdv-dev/RDT/issues/130) by @csala

## 0.2.6 - 2020-10-05

* Add GaussianCopulaTransformer - Issue [#125](https://github.com/sdv-dev/RDT/issues/125) by @csala
* dtype category error - Issue [#124](https://github.com/sdv-dev/RDT/issues/124) by @csala

## 0.2.5 - 2020-09-18

Miunor bugfixing release.

# Bugs Fixed

* Handle NaNs in OneHotEncodingTransformer - Issue [#118](https://github.com/sdv-dev/RDT/issues/118) by @csala
* OneHotEncodingTransformer fails if there is only one category - Issue [#119](https://github.com/sdv-dev/RDT/issues/119) by @csala
* All NaN column produces NaN values enhancement - Issue [#121](https://github.com/sdv-dev/RDT/issues/121) by @csala
* Make the CategoricalTransformer learn the column dtype and restore it back - Issue [#122](https://github.com/sdv-dev/RDT/issues/122) by @csala

## 0.2.4 - 2020-08-08

### General Improvements

* Support Python 3.8 - Issue [#117](https://github.com/sdv-dev/RDT/issues/117) by @csala
* Support pandas >1 - Issue [#116](https://github.com/sdv-dev/RDT/issues/116) by @csala

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
