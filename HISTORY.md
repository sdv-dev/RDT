# History

## 0.5.3 - 2021-10-07

This release fixes a bug with learning rounding digits in the `NumericalTransformer`,
and includes a few housekeeping improvements.

### Issues closed

* Update learn rounding digits to handle all nan data - Issue [#244](https://github.com/sdv-dev/RDT/issues/244) by @katxiao
* Adapt to latest PyLint housekeeping - Issue [#216](https://github.com/sdv-dev/RDT/issues/216) by @fealho

## 0.5.2 - 2021-08-16

This release fixes a couple of bugs introduced by the previous release regarding the
`OneHotEncoder` and the `BooleanTransformer`.

### Issues closed

* BooleanTransformer.reverse_transform sometimes crashes with TypeError - Issue [#210](https://github.com/sdv-dev/RDT/issues/210) by @katxiao
* OneHotEncoder causing shape misalignment in CopulaGAN, CTGAN, and TVAE - Issue [#208](https://github.com/sdv-dev/RDT/issues/208) by @sarahmish
* Boolean.transformer.reverse_transform modifies the input data - Issue [#211](https://github.com/sdv-dev/RDT/issues/211) by @katxiao

## 0.5.1 - 2021-08-11

This release improves the overall performance of the library, both in terms of memory and time consumption.
More specifically, it makes the following modules more efficient: `NullTransformer`, `DatetimeTransformer`,
`LabelEncodingTransformer`, `NumericalTransformer`, `CategoricalTransformer`, `BooleanTransformer` and `OneHotEncodingTransformer`.

It also adds performance-based testing and a script for profiling the performance.

### Issues closed

* Add performance-based testing - Issue [#194](https://github.com/sdv-dev/RDT/issues/194) by @amontanez24
* Audit the NullTransformer - Issue [#192](https://github.com/sdv-dev/RDT/issues/192) by @amontanez24
* Audit DatetimeTransformer - Issue [#189](https://github.com/sdv-dev/RDT/issues/189) by @sarahmish
* Audit the LabelEncodingTransformer - Issue [#184](https://github.com/sdv-dev/RDT/issues/184) by @amontanez24
* Audit the NumericalTransformer - Issue [#181](https://github.com/sdv-dev/RDT/issues/181) by @fealho
* Audit CategoricalTransformer - Issue [#180](https://github.com/sdv-dev/RDT/issues/180) by @katxiao
* Audit BooleanTransformer - Issue [#179](https://github.com/sdv-dev/RDT/issues/179) by @katxiao
* Auditing OneHotEncodingTransformer - Issue [#178](https://github.com/sdv-dev/RDT/issues/178) by @sarahmish
* Create script for profiling - Issue [#176](https://github.com/sdv-dev/RDT/issues/176) by @amontanez24
* Create folder structure for performance testing - Issue [#174](https://github.com/sdv-dev/RDT/issues/174) by @amontanez24

## 0.5.0 - 2021-07-12

This release updates the `NumericalTransformer` by adding a new `rounding` argument.
Users can now obtain numerical values with precision, either pre-specified or automatically computed from the given data.

### Issues closed

* Add `rounding` argument to `NumericalTransformer` - Issue [#166](https://github.com/sdv-dev/RDT/issues/166) by @amontanez24 and @csala
* `NumericalTransformer` rounding error with infinity - Issue [#169](https://github.com/sdv-dev/RDT/issues/169) by @amontanez24
* Add min and max arguments to NumericalTransformer - Issue [#106](https://github.com/sdv-dev/RDT/issues/106) by @amontanez24

## 0.4.2 - 2021-06-08

This release adds a new method to the `CategoricalTransformer` to solve a bug where
the transformer becomes unusable after being pickled and unpickled if it had `NaN`
values in the data which it was fit on.

It also fixes some grammar mistakes in the documentation.

### Issues closed

* CategoricalTransformer with NaN values cannot be pickled bug - Issue [#164](https://github.com/sdv-dev/RDT/issues/164) by @pvk-developer and @csala

### Documentation changes

* docs: fix typo - PR [#163](https://github.com/sdv-dev/RDT/issues/163) by @sbrugman

## 0.4.1 - 2021-03-29

This release improves the `HyperTransformer` memory usage when working with a
high number of columns or a high number of categorical values when using one hot encoding.

### Issues closed

* `Boolean`, `Datetime` and `LabelEncoding` transformers fail with 2D `ndarray` - Issue [#160](https://github.com/sdv-dev/RDT/issues/160) by @pvk-developer
* `HyperTransformer`: Memory usage increase when `reverse_transform` is called - Issue [#156](https://github.com/sdv-dev/RDT/issues/152) by @pvk-developer and @AnupamaGangadhar

## 0.4.0 - 2021-02-24

In this release a change in the HyperTransformer allows using it to transform and
reverse transform a subset of the columns seen during training.

The anonymization functionality which was deprecated and not being used has also
been removed along with the Faker dependency.

### Issues closed

* Allow the HyperTransformer to be used on a subset of the columns - Issue [#152](https://github.com/sdv-dev/RDT/issues/152) by @csala
* Remove faker - Issue [#150](https://github.com/sdv-dev/RDT/issues/150) by @csala

## 0.3.0 - 2021-01-27

This release changes the behavior of the `HyperTransformer` to prevent it from
modifying any column in the given `DataFrame` if the `transformers` dictionary
is passed empty.

### Issues closed

* If transformers is an empty dict, do nothing - Issue [#149](https://github.com/sdv-dev/RDT/issues/149) by @csala

## 0.2.10 - 2020-12-18

This release adds a new argument to the `HyperTransformer` which gives control over
which transformers to use by default for each `dtype` if no specific transformer
has been specified for the field.

This is also the first version to be officially released on conda.

### Issues closed

* Add `dtype_transformers` argument to HyperTransformer - Issue [#148](https://github.com/sdv-dev/RDT/issues/148) by @csala
* Makes Copulas an optional dependency - Issue [#144](https://github.com/sdv-dev/RDT/issues/144) by @fealho

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
