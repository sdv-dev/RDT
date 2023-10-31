# History

## 1.8.0 - 2023-10-31

This release adds the 'random' missing value replacement strategy, which uses random values of the dataset to fill in missing values. 
Additionally users are now able to use the `UniformUnivariate` distribution within the Gaussian Normalizer with this update.

This release contains fixes for the `ClusterBasedNormalizer` which crashes in the reverse transform caused by values being out of bounds 
and a patch for the randomization issue dealing with different values after applying `reset_randomization`.

Anonymization has been moved into RDT library from SDV as it was found to self contained module for RDT and would reduce dependencies needed in SDV.

### Features

* Make the default missing value imputation 'mean' - Issue[#730](https://github.com/sdv-dev/RDT/issues/730) by @R-Palazzo
* When no rounding scheme is detected, log the info instead of showing a warning - Issue[#709](https://github.com/sdv-dev/RDT/issues/709) by @frances-h
* The GaussianNormalizer should accept distribution names that are consistent with scipy - Issue[#656](https://github.com/sdv-dev/RDT/issues/656) by @fealho
* The GaussianNormalizer should accept uniform distributions - Issue[#655](https://github.com/sdv-dev/RDT/issues/655) by @fealho
* Remove psutil - Issue[#615](https://github.com/sdv-dev/RDT/issues/615) by @fealho
* Consider deprecating the FrequencyEncoder - Issue[#614](https://github.com/sdv-dev/RDT/issues/614) by @fealho
* Replace missing values with variable (random) values from the dataset - Issue[#606](https://github.com/sdv-dev/RDT/issues/606)

### Bugs

* RDT Uniform Encoder creates nan Value bug - Issue[#719](https://github.com/sdv-dev/RDT/issues/719) by @lajohn4747
* HyperTransformer transforms while fitting and messes up the random seed - Issue[#716](https://github.com/sdv-dev/RDT/issues/716) by @pvk-developer
* Resolve locales warning for specific sdtype/locale combos (eg. en_US with postcode) - Issue[#701](https://github.com/sdv-dev/RDT/issues/701) by @pvk-developer
* The OrderedLabelEncoder should not accept duplicate categories - Issue[#673](https://github.com/sdv-dev/RDT/issues/673) by @frances-h 
* ClusterBasedNormalizer crashes on reverse transform (IndexError) - Issue[#672](https://github.com/sdv-dev/RDT/issues/672) by @fealho
* Unnecessary warning in OneHotEncoder when there are nan values - Issue[#616](https://github.com/sdv-dev/RDT/issues/616) by @fealho

### Maintenance

* Remove performance tests - Issue[#707](https://github.com/sdv-dev/RDT/issues/707) by @fealho
* ClusterBasedNormalizer code cleanup - Issue[#696](https://github.com/sdv-dev/RDT/issues/696) by @fealho
* Switch default branch from master to main - Issue[#687](https://github.com/sdv-dev/RDT/issues/687) by @amontanez24

### Deprecations

* The `frequencyEncoder` transformer will no longer be supported in future versions of RDT. Please use the `UniformEncoder` transformer instead.
* `GaussianNormalizer` distribution option names have been updated to be consistent with scipy. `gaussian` -> `norm`, `student_t`-> `t`, and `truncated_gaussian` -> `truncnorm`

## 1.7.0 - 2023-08-22

This release adds 3 new transformers:

1. `UniformEncoder` - A categorical and boolean transformer that converts the column into a uniform distribution.
2. `OrderedUniformEncoder` - The same as above, but the order for the categories can be specified, changing which range in the uniform distribution each category belongs to.
3. `IDGenerator`- A text transformer that drops the input column during transform and returns IDs during reverse transform. The IDs all take the form \<prefix>\<number>\<suffix> and can be configured with a custom prefix, suffix and starting point.

Additionally, the `AnonymizedFaker` is enhanced to support the text sdtype. 

### Deprecations

* The `get_input_sdtype` method is being deprecated in favor of `get_supported_sdtypes`.

### New Features

* Create IDGenerator transformer - Issue [#675](https://github.com/sdv-dev/RDT/issues/675) by @R-Palazzo
* Add UniformEncoder (and its ordered version) - Issue [#678](https://github.com/sdv-dev/RDT/issues/678) by @R-Palazzo
* Allow me to use AnonymizedFaker with sdtype text columns - Issue [#688](https://github.com/sdv-dev/RDT/issues/688) by @amontanez24

### Maintenance

* Deprecate get_input_sdtype - Issue [#682](https://github.com/sdv-dev/RDT/issues/682) by @R-Palazzo

## 1.6.1 - 2023-08-02

This release updates the default transformers used for certain sdtypes. It also enables the `AnonymizedFaker` and `PseudoAnonymizedFaker` to work with any sdtype besides boolean, categorical, datetime, numerical or text.

### Bugs

* [Enterprise Usage] Unable to assign generic PII transformers (eg. AnonymizedFaker) - Issue [#674](https://github.com/sdv-dev/RDT/issues/674) by @amontanez24

### New Features

* Update the default transformers that HyperTransformer assigns to each sdtype - Issue [#664](https://github.com/sdv-dev/RDT/issues/664) by @amontanez24

## 1.6.0 - 2023-07-12

This release adds the ability to generate missing values to the `AnonymizedFaker`. Users can now provide the `missing_value_generation` parameter during initialization. They can set it to `None` to not generate any missing values, or `'random'` to generate random missing values in the same proportion as the fitted data.

Additionally, this release improves the `NullTransformer` by allowing nulls to be replaced on the forward transform even if `missing_value_generation` is set to None. It also fixes a bug that was causing the `UnixTimestampEncoder` to return a different dtype than the input on `reverse_transform`. This was particularly problematic when datetime columns are represented as ints.

### New Features

* AnonymizedFaker should be able to model and generate missing values - Issue [#660](https://github.com/sdv-dev/RDT/issues/660) by @R-Palazzo

### Bugs

* The datetime transformers don't give me back the same dtype sometimes - Issue [#657](https://github.com/sdv-dev/RDT/issues/657) by @frances-h
* RDT NullTransformer doesn't replace nulls if missing_value_generation is None - Issue [#658](https://github.com/sdv-dev/RDT/issues/658) by @amontanez24

### Maintenance

* Remove python 3.7 builds - Issue [#663](https://github.com/sdv-dev/RDT/issues/663) by @amontanez24
* Drop support for Python 3.7 - Issue [#666](https://github.com/sdv-dev/RDT/issues/666) by @amontanez24

### Internal

* Add add-on modules to sys.modules - Issue [#653](https://github.com/sdv-dev/RDT/issues/653) by @amontanez24

## 1.5.0 - 2023-06-01

This release adds a new parameter called `missing_value_generation` to the initialization of certain transformers to specify how missing values should be created. The parameter can be used in the `FloatFormatter`, `BinaryEncoder`, `UnixTimestampEncoder`, `OptimizedTimestampEncoder`, `GaussianNormalizer` and `ClusterBasedNormalizer`. Additionally, it fixes a bug that was causing every column that had nulls to generate them in the same place.

### Deprecations

* The `model_missing_values` parameter is being deprecated in favor of the new `missing_value_generation` parameter.

### Bugs

* Fix randomization when creating null values - Issue [#639](https://github.com/sdv-dev/RDT/issues/639) by @fealho

### New Features

* Allow a no-op handling strategy for missing values (nulls) - Issue [#644](https://github.com/sdv-dev/RDT/issues/644) by @pvk-developer
* Add add-on detection for premium transformers - Issue [#646](https://github.com/sdv-dev/RDT/issues/646) by @frances-h

### Maintenance

* Performance tests still fragile - Issue [#641](https://github.com/sdv-dev/RDT/issues/641) by @fealho
* Investigate removing quality tests - Issue [#642](https://github.com/sdv-dev/RDT/issues/642) by @amontanez24

## 1.4.2 - 2023-05-02

This release fixes a bug that caused datetime and numerical transformers to crash if a column was all NaNs. Additionally, it adds support for Pandas 2.0!

### Bugs

* Numerical & datetime transformers crash if the entire column is null - Issue [#637](https://github.com/sdv-dev/RDT/issues/637) by @fraces-h

### Maintenance

* Remove upper bound for pandas - Issue [#633](https://github.com/sdv-dev/RDT/issues/633) by @pvk-developer

## 1.4.1 - 2023-04-25

This release patches an issue that prevented the `RegexGenerator` from working with regexes that had a very large number of possible combinations.

### Bugs

* RegexGenerator continues to have problems if there are too many possibilities - Issue [#635](https://github.com/sdv-dev/RDT/issues/635) by @pvk-developer

## 1.4.0 - 2023-04-13

This release adds a couple of new features including adding the `OrderedLabelEncoder` and deprecating the `CustomLabelEncoder`. It also adds a change that makes all generator type transformers in the `HyperTransformer` use a different random seed.

Additionally, bugs were patched in the `RegexGenerator` that caused it to crash or take too long in certain cases. Finally, this release improved the detection of Faker functions in the `AnonymizedFaker`.

### Bugs

* Find nested Faker provider submodules - PR [#630](https://github.com/sdv-dev/RDT/pull/630) by @frances-h
* RegexGenerator fails to generate values if there are too many possibilities - Issue [#623](https://github.com/sdv-dev/RDT/issues/623) by @R-Palazzo
* RegexGenerator takes too much time and runs out of memory if there are too many possibilities - Issue [#624](https://github.com/sdv-dev/RDT/issues/624) by @R-Palazzo

### New Features

* Choose a different seed for each transformer - Issue [#619](https://github.com/sdv-dev/RDT/issues/619) by @fealho
* Rename CustomLabelEncoder to OrderedLabelEncoder - Issue [#621](https://github.com/sdv-dev/RDT/issues/621) by @R-Palazzo
* Add functionality to find version add-on - Issue [#620](https://github.com/sdv-dev/RDT/issues/620) by @frances-h

## 1.3.0 - 2023-01-18

This release makes changes to the way that individual transformers are stored in the `HyperTransformer`. When accessing the config via `HyperTransformer.get_config()`, the transformers listed in the config are now the actual transformer instances used during fitting and transforming. These instances can now be accessed and used to examine their properties post fitting. For example, you can now view the mapping for a `PseudoAnonymizedFaker` instance using `PseudoAnonymizedFaker.get_mapping()` on the instance retrieved from the config.

Additionally, the output of `reverse_tranform` no longer appends the `.value` suffix to every unnamed output column. Only output columns that are created from context extracted from the input columns will have suffixes (eg. `.normalized` in the `ClusterBasedNormalizer`).

The `AnonymizedFaker` and `RegexGenerator` now have an `enforce_uniqueness` parameter, which controls whether the data returned by `reverse_transform` should be unique. The `HyperTransformer` now has a method called `create_anonymized_columns` that can be used to generate columns that are matched with anonymizing transformers like `AnonymizedFaker` and `RegexGenerator`. The method can be used as follows:
`HyperTransformer.create_anonymized_columns(num_rows=5, column_names=['email_optin', 'credit_card'])`

Another major change in this release is the ability to control randomization. Every time a `HyperTransformer` is initialized, its randomness will be reset to the same seed, and it will yield the same results for `reverse_transform` if given the same input. Every subsequent call to `reverse_transform` yields a different result. If a user desires to reset the seed, they can call `HyperTransformer.reset_randomization`.

Finally, this release adds support for Python 3.10 and drops support for 3.6.

### Bugs

* The reset_randomization should also apply to fit and transform - Issue [#608](https://github.com/sdv-dev/RDT/issues/608) by @amontanez24
* Cannot print CustomLabelEncoder: ValueError - Issue [#607](https://github.com/sdv-dev/RDT/issues/607) by @amontanez24
* Float formatter learn_rounding_scheme doesn't work on all digits - Issue [#556](https://github.com/sdv-dev/RDT/issues/556) by @fealho
* Warnings not showing on update_transformers_by_sdtype - Issue [#582](https://github.com/sdv-dev/RDT/issues/582) by @amontanez24
* OneHotEncoder doesn't work with boolean sdtype - Issue [#583](https://github.com/sdv-dev/RDT/issues/583) by @pvk-developer
* Setting config on HyperTransformer does not read supported_sdtypes - Issue [#560](https://github.com/sdv-dev/RDT/issues/560) by @pvk-developer
* https://github.com/sdv-dev/RDT/issues/545 - Issue [#545](https://github.com/sdv-dev/RDT/issues/545) by @pvk-developer
* Add error to NullTransformer when data only contains nans - PR [#567](https://github.com/sdv-dev/RDT/pull/567) by @fealho
* Update update_transformers validation - PR [#563](https://github.com/sdv-dev/RDT/pull/563) by @fealho

### Maintenance

* Support Python 3.10 - Issue [#593](https://github.com/sdv-dev/RDT/issues/593) by @pvk-developer
* RDT 1.3 Package Maintenance Updates - Issue [#594](https://github.com/sdv-dev/RDT/issues/594) by @pvk-developer

### New Features

* Update errors - Issue [#599](https://github.com/sdv-dev/RDT/issues/599) by @amontanez24
* Add ability to control randomness - Issue [#584](https://github.com/sdv-dev/RDT/issues/584) by @amontanez24
* Printing and error improvements - Issue [#581](https://github.com/sdv-dev/RDT/issues/581) by @amontanez24
* Make RegexGenerator not to reset itself - Issue [#558](https://github.com/sdv-dev/RDT/issues/558) by @pvk-developer
* Add a reset_anonymization method - Issue [#559](https://github.com/sdv-dev/RDT/issues/559) by @pvk-developer
* Don't copy instances of tranformer - Issue [#541](https://github.com/sdv-dev/RDT/issues/541) by @fealho
* Remove '.value' suffix - Issue [#533](https://github.com/sdv-dev/RDT/issues/533) by @fealho
* Change the NEXT_TRANSFORMERS logic - Issue [#557](https://github.com/sdv-dev/RDT/issues/557) by @fealho
* Add utility functions to AnonymizedFaker - Issue [#561](https://github.com/sdv-dev/RDT/issues/561) by @pvk-developer
* Update API for update_transformers_by_sdtype to be more explicit about instances vs. copies - Issue [#540](https://github.com/sdv-dev/RDT/issues/540) by @fealho
* Add create_anonymized_columns method to anonymize data from scratch - Issue [#546](https://github.com/sdv-dev/RDT/issues/546) by @pvk-developer
* Add parameter to AnonymizedFaker() and RegexGenerator() to generate only unique values - Issue [#542](https://github.com/sdv-dev/RDT/issues/542) by @pvk-developer

## 1.2.1 - 2022-9-12

This release fixes a bug that caused the `UnixTimestampEncoder` to return data with the incorrect datetime format. It also fixes a bug that caused the null column
not to be reverse transformed when using the `UnixTimestampEncoder` when the `missing_value_replacement` was not set.

### Bugs

* Inconsistency in date format after reverse transform - Issue [#515](https://github.com/sdv-dev/RDT/issues/515) by @pvk-developer
* Fix calling null_transformer with model_missing_values. - PR [#550](https://github.com/sdv-dev/RDT/pull/550) by @pvk-developer

## 1.2.0 - 2022-8-17

This release adds a new transformer called the `PseudoAnonymizedFaker`. This transformer enables the pseudo-anonymization of your data by mapping all of a column's original values to fake values that get returned during the reverse transformation process. Each original value is always mapped to the same fake value.

Additionally, this release enables the `HyperTransformer` to use categorical transformers on boolean columns. It also introduces a new parameter called `computer_representation` to the `FloatFormatter` that will allow for values to be clipped to certain bounds based on the computer type used for a numerical column.

Finally, this release patches a bug that caused unpredicatable results from the `reverse_transform` method of the `FrequencyEncoder` when `add_noise` is enabled.

### New Features

* Add PseudoAnonymizedFaker transformer - Issue [#517](https://github.com/sdv-dev/RDT/issues/517) by @pvk-developer
* Boolean columns should be able to use any of the categorical transformers - Issue[#527](https://github.com/sdv-dev/RDT/issues/527) by @pvk-developer
* Update FloatFormatter with parameters for the computer representation - Issue[#521](https://github.com/sdv-dev/RDT/issues/521) by @fealho

### Bugs

* Unpredictable results for FrequencyEncoder(add_noise=True) - Issue [#528](https://github.com/sdv-dev/RDT/issues/528) by @fealho

### Internal

* Performance Tests update - Issue [#524](https://github.com/sdv-dev/RDT/issues/524) by @pvk-developer

## 1.1.0 - 2022-6-9

This release adds multiple new transformers: the `CustomLabelEncoder` and the `RegexGenerator`. The `CustomLabelEncoder` works similarly 
to the `LabelEncoder`, except it allows users to provide the order of the categories. The `RegexGenerator` allows users to specify a regex
pattern and will generate values that match that pattern.

This release also improves current transformers. The `LabelEncoder` now has a parameter called `order_by` that allows users to specify the 
ordering scheme for their data (eg. order numerically or alphabetically). The `LabelEncoder` also now has a parameter called `add_noise` 
that allows users to specify whether or not uniform noise should be added to the transformed data. Performance enhancements were made for the 
`GaussianNormalizer` by removing an unnecessary distribution search and the `FloatFormatter` will no longer round values to any place higher 
than the ones place by default.

### New Features

* Add noise parameter to LabelEncoder - Issue [#500](https://github.com/sdv-dev/RDT/issues/500) by @fealho
* Remove parameters related to distribution search and change default for GaussianNormalizer - Issue [#499](https://github.com/sdv-dev/RDT/issues/499) 
by @amontanez24
* Add order_by parameter to LabelEncoder - Issue [#510](https://github.com/sdv-dev/RDT/issues/506) by @amontanez24
* Only round to decimal places in FloatFormatter - Issue [#508](https://github.com/sdv-dev/RDT/issues/508) by @fealho
* Add CustomLabelEncoder transformer - Issue [#507](https://github.com/sdv-dev/RDT/issues/507) by @amontanez24
* Add RegexGenerator Transformer - Issue [#505](https://github.com/sdv-dev/RDT/issues/505) by @pvk-developer

## 1.0.0 - 2022-4-25

The main update of this release is the introduction of a `config`, which describes the `sdtypes` and `transformers` that will be used by the `HyperTransformer` for each column of the data, where `sdtype` stands for the **semantic** or **statistical** meaning of a datatype. The user can interact with this config through the newly created methods `update_sdtypes`, `get_config`, `set_config`, `update_transformers`, `update_transformers_by_sdtype` and `remove_transformer_by_sdtype`.

This release also included various new features and updates, including:

* Users can now transform subsets of the data using its own methods, `transform_subset` and `reverse_transform_subset`.
* User validation was added for the following methods: `transform`, `reverse_transform`, `update_sdtypes`, `update_transformers`, `set_config`.
* Unnecessary warnings were removed from `GaussianNormalizer.fit` and `FrequencyEncoder.transform`.
* The user can now set a transformers as None.
* Transformers that cannot work with missing values will automatically fill them in.
* Added support for additional datetime formats.
* Setting  `model_missing_values = False` in a transformer was updated to keep track of the percentage of missing values, instead of producing data containing `NaN`'s.
* All parameters were removed from the `HyperTransformer`.
* The demo dataset `get_demo` was improved to be more intuitive.

Finally, a number of transformers were redesigned to be more user friendly. Among them, the following transformers have also been renamed:

* `BayesGMMTransformer` -> `ClusterBasedNormalizer`
* `GaussianCopulaTransformer` -> `GaussianNormalizer`
* `DateTimeRoundedTransformer` -> `OptimizedTimestampEncoder`
* `DateTimeTransformer` -> `UnixTimestampEncoder`
* `NumericalTransformer` -> `FloatFormatter`
* `LabelEncodingTransformer` -> `LabelEncoder`
* `OneHotEncodingTransformer` -> `OneHotEncoder`
* `CategoricalTransformer` -> `FrequencyEncoder`
* `BooleanTransformer` -> `BinaryEncoder`
* `PIIAnonymizer` -> `AnonymizedFaker`

### New Features

* Fix using None as transformer when update_transformers_by_sdtype - Issue [#496](https://github.com/sdv-dev/RDT/issues/496) by @pvk-developer
* Rename PIIAnonymizer --> AnonymizedFaker - Issue [#483](https://github.com/sdv-dev/RDT/issues/483) by @pvk-developer
* User validation for reverse_transform - Issue [#480](https://github.com/sdv-dev/RDT/issues/480) by @amontanez24
* User validation for transform - Issue [#479](https://github.com/sdv-dev/RDT/issues/479) by @fealho\
* User validation for set_config - Issue [#478](https://github.com/sdv-dev/RDT/issues/478) by @fealho
* User validation for update_transformers_by_sdtype - Issue [#477](https://github.com/sdv-dev/RDT/issues/477) by @amontanez24
* User validation for update_transformers - Issue [#475](https://github.com/sdv-dev/RDT/issues/475) by @fealho
* User validation for update_sdtypes - Issue [#474](https://github.com/sdv-dev/RDT/issues/474) by @fealho
* Allow columns to not have a transformer - Issue [#473](https://github.com/sdv-dev/RDT/issues/473) by @pvk-developer
* Create methods to transform a subset of the data (& reverse transform it) - Issue [#472](https://github.com/sdv-dev/RDT/issues/472) by @amontanez24
* Throw a warning if you use set_config on a HyperTransformer that's already fit - Issue [#466](https://github.com/sdv-dev/RDT/issues/466) by @amontanez24
* Update README for RDT 1.0 - Issue [#454](https://github.com/sdv-dev/RDT/issues/454) by @amontanez24
* Issue with printing PIIAnonymizer in HyperTransformer - Issue [#452](https://github.com/sdv-dev/RDT/issues/452) by @pvk-developer
* Pretty print get_config - Issue [#450](https://github.com/sdv-dev/RDT/issues/450) by @pvk-developer
* Silence warning for GaussianNormalizer.fit - Issue [#443](https://github.com/sdv-dev/RDT/issues/443) by @pvk-developer
* Transformers that cannot work with missing values should automatically fill them in - Issue [#442](https://github.com/sdv-dev/RDT/issues/442) by @amontanez24
* More descriptive error message in PIIAnonymizer when provider_name and function_name don't align - Issue [#440](https://github.com/sdv-dev/RDT/issues/440) by @pvk-developer
* Can we support additional datetime formats? - Issue [#439](https://github.com/sdv-dev/RDT/issues/439) by @pvk-developer
* Update FrequencyEncoder.transform so that pandas won't throw a warning - Issue [#436](https://github.com/sdv-dev/RDT/issues/436) by @pvk-developer
* Update functionality when model_missing_values=False - Issue [#435](https://github.com/sdv-dev/RDT/issues/435) by @amontanez24
* Create methods for getting and setting a config - Issue [#418](https://github.com/sdv-dev/RDT/issues/418) by @amontanez24
* Input validation & error handling in HyperTransformer - Issue [#408](https://github.com/sdv-dev/RDT/issues/408) by @fealho and @amontanez24
* Remove unneeded params from HyperTransformer - Issue [#407](https://github.com/sdv-dev/RDT/issues/407) by @pvk-developer
* Rename property: _valid_output_sdtypes - Issue [#406](https://github.com/sdv-dev/RDT/issues/406) by @amontanez24
* Add pii as a new sdtype in HyperTransformer - Issue [#404](https://github.com/sdv-dev/RDT/issues/404) by @pvk-developer
* Update transformers by data type (in HyperTransformer) - Issue [#403](https://github.com/sdv-dev/RDT/issues/403) by @pvk-developer
* Update transformers by column name in HyperTransformer - Issue [#402](https://github.com/sdv-dev/RDT/issues/402) by @pvk-developer
* Improve updating field_data_types in HyperTransformer - Issue [#400](https://github.com/sdv-dev/RDT/issues/400) by @amontanez24
* Create method to auto detect HyperTransformer config from data - Issue [#399](https://github.com/sdv-dev/RDT/issues/399) by @fealho
* Update HyperTransformer default transformers - Issue [#398](https://github.com/sdv-dev/RDT/issues/398) by @fealho
* Add PIIAnonymizer - Issue [#397](https://github.com/sdv-dev/RDT/issues/397) by @pvk-developer
* Improve the way we print an individual transformer - Issue [#395](https://github.com/sdv-dev/RDT/issues/395) by @amontanez24
* Rename columns parameter in fit for each individual transformer - Issue [#376](https://github.com/sdv-dev/RDT/issues/376) by @fealho and @pvk-developer
* Create a more descriptive demo dataset - Issue [#374](https://github.com/sdv-dev/RDT/issues/374) by @fealho
* Delete unnecessary transformers - Issue [#373](https://github.com/sdv-dev/RDT/issues/373) by @fealho
* Update NullTransformer to make it user friendly - Issue [#372](https://github.com/sdv-dev/RDT/issues/372) by @pvk-developer
* Update BayesGMMTransformer to make it user friendly - Issue [#371](https://github.com/sdv-dev/RDT/issues/371) by @amontanez24
* Update GaussianCopulaTransformer to make it user friendly - Issue [#370](https://github.com/sdv-dev/RDT/issues/370) by @amontanez24
* Update DateTimeRoundedTransformer to make it user friendly - Issue [#369](https://github.com/sdv-dev/RDT/issues/369) by @amontanez24
* Update DateTimeTransformer to make it user friendly - Issue [#368](https://github.com/sdv-dev/RDT/issues/368) by @amontanez24
* Update NumericalTransformer to make it user friendly - Issue [#367](https://github.com/sdv-dev/RDT/issues/367) by @amontanez24
* Update LabelEncodingTransformer to make it user friendly - Issue [#366](https://github.com/sdv-dev/RDT/issues/366) by @fealho
* Update OneHotEncodingTransformer to make it user friendly - Issue [#365](https://github.com/sdv-dev/RDT/issues/365) by @fealho
* Update CategoricalTransformer to make it user friendly - Issue [#364](https://github.com/sdv-dev/RDT/issues/364) by @fealho
* Update BooleanTransformer to make it user friendly - Issue [#363](https://github.com/sdv-dev/RDT/issues/363) by @fealho
* Update names & functionality for handling missing values - Issue [#362](https://github.com/sdv-dev/RDT/issues/362) by @pvk-developer

### Bugs

* Checking keys of config as set - Issue [#497](https://github.com/sdv-dev/RDT/issues/497) by @amontanez24
* Only update transformer used when necessary for update_sdtypes - Issue [#469](https://github.com/sdv-dev/RDT/issues/469) by @amontanez24
* Fix how get_config prints transformers - Issue [#468](https://github.com/sdv-dev/RDT/issues/468) by @pvk-developer
* NullTransformer reverse_transform alters input data due to not copying - Issue [#455](https://github.com/sdv-dev/RDT/issues/455) by @amontanez24
* Attempting to transform a subset of the data should lead to an Error - Issue [#451](https://github.com/sdv-dev/RDT/issues/451) by @amontanez24
* Detect_initial_config isn't detecting sdtype "numerical" - Issue [#449](https://github.com/sdv-dev/RDT/issues/449) by @pvk-developer
* PIIAnonymizer not generating multiple locales - Issue [#447](https://github.com/sdv-dev/RDT/issues/447) by @pvk-developer
* Error when printing ClusterBasedNormalizer and GaussianNormalizer - Issue [#441](https://github.com/sdv-dev/RDT/issues/441) by @pvk-developer
* Datetime reverse transform crashes if datetime_format is specified - Issue [#438](https://github.com/sdv-dev/RDT/issues/438) by @amontanez24
* Correct datetime format is not recovered on reverse_transform - Issue [#437](https://github.com/sdv-dev/RDT/issues/437) by @pvk-developer
* Use numpy NaN values in BinaryEncoder - Issue [#434](https://github.com/sdv-dev/RDT/issues/434) by @pvk-developer
* Duplicate _output_columns during fitting - Issue [#423](https://github.com/sdv-dev/RDT/issues/423) by @fealho

### Internal Improvements

* Making methods that aren't part of API private - Issue [#489](https://github.com/sdv-dev/RDT/issues/489) by @amontanez24
* Fix columns missing in config and update transformers to None - Issue [#495](https://github.com/sdv-dev/RDT/issues/495) by @pvk-developer

## 0.6.4 - 2022-3-7

This release fixes multiple bugs concerning the `HyperTransformer`. One is that the `get_transformer_tree_yaml` method no longer crashes on
every call. Another is that calling the `update_field_data_types` and `update_default_data_type_transformers` after fitting no longer breaks the `transform`
method.

The `HyperTransformer` now sorts its outputs for both `transform` and `reverse_transform` based on the order of the input's columns. It is also now possible
to create transformers that simply drops columns during `transform` and don't return any new columns.

### New Features

* Support dropping a column trough a transformer - Issue [#393](https://github.com/sdv-dev/RDT/issues/393) by @pvk-developer
* HyperTransformer should sort columns after transform and reverse_transform - Issue [#405](https://github.com/sdv-dev/RDT/issues/405) by @fealho

### Bugs

* get_transformer_tree_yaml fails - Issue [#389](https://github.com/sdv-dev/RDT/issues/389) by @amontanez24
* HyperTransformer _unfit method not working correctly - Issue [#390](https://github.com/sdv-dev/RDT/issues/390) by @amontanez24
* Blank dataframe after updating the data types - Issue [#401](https://github.com/sdv-dev/RDT/issues/401) by @amontanez24

## 0.6.3 - 2022-2-4

This release adds a new module to the `RDT` library called `performance`. This module can be used to evaluate the speed and peak memory usage
of any transformer in RDT. This release also increases the maximum acceptable version of scikit-learn to make it more compatible with other libraries
in the `SDV` ecosystem. On top of that, it fixes a bug related to a new version of `pandas`.

### New Features

* Move profiling functions into RDT library - Issue [#353](https://github.com/sdv-dev/RDT/issues/353) by @amontanez24

### Housekeeping

* Increase scikit-learn dependency range - Issue [#351](https://github.com/sdv-dev/RDT/issues/351) by @amontanez24
* pandas 1.4.0 release causes a small error - Issue [#358](https://github.com/sdv-dev/RDT/issues/358) by @fealho

### Bugs

* Performance tests get stuck on Unix if multiprocessing is involved - Issue [#337](https://github.com/sdv-dev/RDT/issues/337) by @amontanez24

## 0.6.2 - 2021-12-28

This release adds a new `BayesGMMTransformer`. This transformer can be used to convert a numerical column into two
columns: a discrete column indicating the selected `component` of the GMM for each row, and a continuous column containing
the normalized value of each row based on the `mean` and `std` of the selected `component`. It is useful when the column being transformed
came from multiple distributions.

This release also adds multiple new methods to the `HyperTransformer` API. These allow for users to access the specfic
transformers used on each input field, as well as view the entire tree of transformers that are used when running `transform`.
The exact methods are:

* `BaseTransformer.get_input_columns()` - Return list of input columns for a transformer.
* `BaseTransformer.get_output_columns()` - Return list of output columns for a transformer.
* `HyperTransformer.get_transformer(field)` - Return the transformer instance used for a field.
* `HyperTransformer.get_output_transformers(field)` - Return dictionary mapping output columns of a field to the transformers used on them.
* `HyperTransformer.get_final_output_columns(field)` - Return list of all final output columns related to a field.
* `HyperTransformer.get_transformer_tree_yaml()` - Return YAML representation of transformers tree.

Additionally, this release fixes a bug where the `HyperTransformer` was incorrectly raising a `NotFittedError`. It also improved the
`DatetimeTransformer` by autonomously detecting if a column needs to be converted from `dtype` `object` to `dtype` `datetime`.

### New Features

* Cast column to datetime if specified in field transformers - Issue [#321](https://github.com/sdv-dev/RDT/issues/321) by @amontanez24
* Add a BayesianGMM Transformer - Issue [#183](https://github.com/sdv-dev/RDT/issues/183) by @fealho
* Add transformer tree structure and traversal methods - Issue [#330](https://github.com/sdv-dev/RDT/issues/330) by @amontanez24

### Bugs fixed

* HyperTransformer raises NotFittedError after fitting - Issue [#332](https://github.com/sdv-dev/RDT/issues/332) by @amontanez24

## 0.6.1 - 2021-11-10

This release adds support for Python 3.9! It also removes unused document files.

### Internal Improvements

* Add support for Python 3.9 - Issue [#323](https://github.com/sdv-dev/RDT/issues/323) by @amontanez24
* Remove docs - PR [#322](https://github.com/sdv-dev/RDT/pull/322) by @pvk-developer

## 0.6.0 - 2021-10-29

This release makes major changes to the underlying code for RDT as well as the API for both the `HyperTransformer` and `BaseTransformer`.
The changes enable the following functionality:

* The `HyperTransformer` can now apply a sequence of transformers to a column.
* Transformers can now take multiple columns as an input.
* RDT has been expanded to allow for infinite data types to be added instead of being restricted to `pandas.dtypes`.
* Users can define acceptable output types for running `HyperTransformer.transform`.
* The `HyperTransformer` will continuously apply transformations to the input fields until only acceptable data types are in the output. 
* Transformers can return data of any data type.
* Transformers now have named outputs and output types.
* Transformers can suggest which transformer to use on any of their outputs.

To take advantage of this functionality, the following API changes were made:

* The `HyperTransformer` has new initialization parameters that allow users to specify data types for any field in their data as well as
specify which transformer to use for a field or data type. The parameters are:
    * `field_transformers` - A dictionary allowing users to specify which transformer to use for a field or derived field. Derived fields
    are fields created by running `transform` on the input data.
    * `field_data_types` - A dictionary allowing users to specify the data type of a field.
    * `default_data_type_transformers` - A dictionary allowing users to specify the default transformer to use for a data type.
    * `transform_output_types` - A dictionary allowing users to specify which data types are acceptable for the output of `transform`.
    This is a result of the fact that transformers can now be applied in a sequence, and not every transformer will return numeric data.
* Methods were also added to the `HyperTransformer` to allow these parameters to be modified. These include `get_field_data_types`,
`update_field_data_types`, `get_default_data_type_transformers`, `update_default_data_type_transformers` and `set_first_transformers_for_fields`.
* The `BaseTransformer` now requires the column names it will transform to be provided to `fit`, `transform` and `reverse_transform`.
* The `BaseTransformer` added the following method to allow for users to see its output fields and output types: `get_output_types`.
* The `BaseTransformer` added the following method to allow for users to see the next suggested transformer for each output field:
`get_next_transformers`. 

On top of the changes to the API and the capabilities of RDT, many automated checks and tests were also added to ensure that contributions
to the library abide by the current code style, stay performant and result in data of a high quality. These tests run on every push to the
repository. They can also be run locally via the following functions:

* `validate_transformer_code_style` - Checks that new code follows the code style.
* `validate_transformer_quality` - Tests that new transformers yield data that maintains relationships between columns.
* `validate_transformer_performance` - Tests that new transformers don't take too much time or memory.
* `validate_transformer_unit_tests` - Checks that the unit tests cover all new code, follow naming conventions and pass.
* `validate_transformer_integration` - Checks that the integration tests follow naming conventions and pass.

### New Features

* Update HyperTransformer API - Issue [#298](https://github.com/sdv-dev/RDT/issues/298) by @amontanez24
* Create validate_pull_request function - Issue [#254](https://github.com/sdv-dev/RDT/issues/254) by @pvk-developer
* Create validate_transformer_unit_tests function - Issue [#249](https://github.com/sdv-dev/RDT/issues/249) by @pvk-developer
* Create validate_transformer_performance function - Issue [#251](https://github.com/sdv-dev/RDT/issues/251) by @katxiao
* Create validate_transformer_quality function - Issue [#253](https://github.com/sdv-dev/RDT/issues/253) by @amontanez24
* Create validate_transformer_code_style function - Issue [#248](https://github.com/sdv-dev/RDT/issues/248) by @pvk-developer
* Create validate_transformer_integration function - Issue [#250](https://github.com/sdv-dev/RDT/issues/250) by @katxiao
* Enable users to specify transformers to use in HyperTransformer - Issue [#233](https://github.com/sdv-dev/RDT/issues/233) by @amontanez24 and @csala
* Addons implementation - Issue [#225](https://github.com/sdv-dev/RDT/issues/225) by @pvk-developer
* Create ways for HyperTransformer to know which transformers to apply to each data type - Issue [#232](https://github.com/sdv-dev/RDT/issues/232) by @amontanez24 and @csala
* Update categorical transformers - PR [#231](https://github.com/sdv-dev/RDT/pull/231) by @fealho
* Update numerical transformer - PR [#227](https://github.com/sdv-dev/RDT/pull/227) by @fealho
* Update datetime transformer - PR [#230](https://github.com/sdv-dev/RDT/pull/230) by @fealho
* Update boolean transformer - PR [#228](https://github.com/sdv-dev/RDT/pull/228) by @fealho
* Update null transformer - PR [#229](https://github.com/sdv-dev/RDT/pull/229) by @fealho
* Update the baseclass - PR [#224](https://github.com/sdv-dev/RDT/pull/224) by @fealho

### Bugs fixed

* If the input data has a different index, the reverse transformed data may be out of order - Issue [#277](https://github.com/sdv-dev/RDT/issues/277) by @amontanez24

### Documentation changes

* RDT contributing guide - Issue [#301](https://github.com/sdv-dev/RDT/issues/301) by @katxiao and @amontanez24

### Internal improvements

* Add PR template for new transformers - Issue [#307](https://github.com/sdv-dev/RDT/issues/307) by @katxiao
* Implement Quality Tests for Transformers - Issue [#252](https://github.com/sdv-dev/RDT/issues/252) by @amontanez24
* Update performance test structure - Issue [#257](https://github.com/sdv-dev/RDT/issues/257) by @katxiao
* Automated integration test for transformers - Issue [#223](https://github.com/sdv-dev/RDT/issues/223) by @katxiao
* Move datasets to its own module - Issue [#235](https://github.com/sdv-dev/RDT/issues/235) by @katxiao
* Fix missing coverage in rdt unit tests - Issue [#219](https://github.com/sdv-dev/RDT/issues/219) by @fealho
* Add repo-wide automation - Issue [#309](https://github.com/sdv-dev/RDT/issues/309) by @katxiao

### Other issues closed

* DeprecationWarning: np.float is a deprecated alias for the builtin float - Issue [#304](https://github.com/sdv-dev/RDT/issues/304) by @csala
* Add pip check to CI workflows - Issue [#290](https://github.com/sdv-dev/RDT/issues/290) by @csala
* Should Transformers subclasses exist for specific configurations? - Issue [#243](https://github.com/sdv-dev/RDT/issues/243) by @fealho

## 0.5.3 - 2021-10-07

This release fixes a bug with learning rounding digits in the `NumericalTransformer`,
and includes a few housekeeping improvements.

### Issues closed

* Update learn rounding digits to handle all nan data - Issue [#244](https://github.com/sdv-dev/RDT/issues/244) by @katxiao
* Adapt to latest PyLint housekeeping - Issue [#216](https://github.com/sdv-dev/RDT/issues/216) by @fealho

## 0.5.2 - 2021-08-16

This release fixes a couple of bugs introduced by the previous release regarding the
`OneHotEncodingTransformer` and the `BooleanTransformer`.

### Issues closed

* BooleanTransformer.reverse_transform sometimes crashes with TypeError - Issue [#210](https://github.com/sdv-dev/RDT/issues/210) by @katxiao
* OneHotEncodingTransformer causing shape misalignment in CopulaGAN, CTGAN, and TVAE - Issue [#208](https://github.com/sdv-dev/RDT/issues/208) by @sarahmish
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
