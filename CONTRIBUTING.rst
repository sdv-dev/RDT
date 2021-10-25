.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

.. contents:: Table of contents
   :local:
   :depth: 3

Contributing to RDT
-------------------

Setting Up a Local Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the `Reversible Data Transforms` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/RDT.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv RDT
    $ cd RDT/
    $ make install-develop

4. Claim or file an issue on GitHub. If there is already an issue on GitHub for the
   contribution you wish to make, claim it. If not, please file an issue and then claim
   it before creating a branch.

5. Create a branch for local development::

    $ git checkout -b issue-[issue-number]-description-of-your-bugfix-or-feature

   The naming scheme for your branch should have a prefix of the format ``issue-X``
   where X is the associated issue's number (eg. ``issue-123-fix-foo-bug``). If you
   are not developing on your own fork, further prefix the branch with your GitHub
   username, like ``githubusername/gh-123-fix-foo-bug``.

   Now you can make your changes locally.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-branch

Code Style
~~~~~~~~~~

RDT follows certain coding style guidelines. Any change made should conform to these
guidelines. RDT using the following third party libraries to check the code style.

* flake8::

    $ flake8 rdt
    $ flake8 tests --ignore=D

* isort::

    $ isort -c --recursive rdt tests

* pylint::

    $ pylint rdt tests/performance --rcfile=setup.cfg

* pydocstyle::

    $ pydocstyle rdt
    $ pydocstyle tests

To run all of the code style checks in RDT, use the following command::

    $ make lint

or if you are developing on Windows you can use::

    $ invoke lint

Unit Test Guidelines
~~~~~~~~~~~~~~~~~~~~

There should be unit tests created specifically for any changes you add.
The unit tests are expected to cover 100% of your contribution's code based on the
coverage report. All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests that cover a module called ``rdt/path/to/a_module.py`` should be implemented
   in a separated module called ``tests/unit/path/to/test_a_module.py.`` Note that the module
   name has the ``test_`` prefix and is located in a path similar to the one of the tested
   module, just inside the tests folder.

3. Each method of the tested module should have at least one associated test method, and
   each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have descriptive names
   that indicate which scenario they cover.
   Names such as ``test_some_method_input_none``, ``test_some_method_value_error`` or
   ``test_some_method_timeout`` are good, but names like ``test_some_method_1``,
   ``some_method`` or ``test_error`` are not.

5. Each test should validate only what the code of the method being tested does, and not
   cover the behavior of any third party package or tool being used, which is assumed to
   work properly as far as it is being passed the right values.

6. Any third party tool that may have any kind of random behavior, such as some Machine
   Learning models, databases or Web APIs, will be mocked using the ``mock`` library, and
   the only thing that will be tested is that our code passes the right values to them.

7. Unit tests should not use anything from outside the test and the code being tested. This
   includes not reading or writing to any file system or database, which will be properly
   mocked.

To run the test suite in RDT locally, use the following command::

    $ make test

or if you are developing on Windows, use::

    $ invoke test

.. _Pull Request Guidelines:

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code.
4. The pull request should work for all the supported Python versions. Check the `Github actions
   page`_ and make sure that all the checks pass.

Contributing a New Transformer
------------------------------

In addition to the guidelines mentioned above, there are extra steps that need to be taken
when adding a new ``Transformer`` class. They are described in detail in this section.

Transformer Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

When contributing a new transformer, the most obvious requirement is creating the new Transformer
class. The class should inherit from `BaseTransformer` or one of its child classes.
There are only three required methods for a transformer:

1. ``_fit(data: pd.DataFrame)``: Used to store and learn any values from the input data that
   might be useful for the transformer.
2. ``_transform(data: pd.DataFrame)``: Used to transform the input data into completely numeric
   data. This method should not modify the internal state of the Transformer instance.
3. ``_reverse_transform(data: pd.DataFrame)``: Used to convert data that is completely numeric
   back into the format of the fitted data. This method should not modify the internal state of
   the Transformer instance.

Each transformer class should be placed inside the ``rdt/transformers`` folder, in a module
file named after the data type that the transformer operates on. For example, if you are
writing a transformer that works with ``categorical`` data, your transformer should be placed
inside the ``rdt/transformers/categorical.py`` module.

For more detailed guide on writing transformers, refer to the `Development Guide`_.

On top of adding the new class, unit tests must be written to cover all of the methods the new
class uses. In some cases, integration tests may also be required. More details on this can be
found below.

If the transformer add a previously unsupported `Data Type` to RDT, then more steps will need
to be taken for the quality and performance tests. A new `DatasetGenerator` class may need to
be created for the `Data Type`. You may also need to find a real world dataset containing this
`Data Type` and request for it to be added. More details for these steps can be found below in
the `Transformer Performance`_ and `Transformer Quality`_ sections respectively.

Transformer Validations
~~~~~~~~~~~~~~~~~~~~~~~

.. _Code Style:

Code Style
""""""""""

The code added for the new transformer must abide by the code style used in RDT. In addition,
there are custom code style requirements that must also be met. These mostly have to do with
class and method naming conventions. For example, all transformer classes must ened in
``Transformer``. They also have to inherit from the ``rdt.transformers.BaseTransformer`` class.

Validating Code Style
*********************

To validate the overall code style for your transformer, you can use the custom code validation
function, ``validate_transformer_code_style``. This function returns a boolean indicating whether
or not the transformer passed all the code style checks. It also prints a table describing each
check and whether or not it passed.

.. code-block:: Python

   In [1]: from tests.contributing import validate_transformer_code_style

   In [2]: valid = validate_transformer_code_style('rdt.transformers.BooleanTransformer') # Replace BooleanTransformer with your transformer
   Validating source file C:\Datacebo\RDT\rdt\transformers\boolean.py

   SUCCESS: The code style is correct.

   Check                      Correct    Details
   -------------------------  ---------  ---------------------------------------------------------
   flake8                     Yes        Code follows PEP8 standards.
   isort                      Yes        Imports are properly sorted.
   pylint                     Yes        Code is properly formatted and structured.
   pydocstyle                 Yes        The docstrings are properly written.
   Transformer Name           Yes        Transformer name ends with ``Transformer``.
   Transformer is subclass    Yes        The transformer is subclass of ``BaseTransformer``.
   Valid module               Yes        The transformer is placed inside a valid module.
   Valid test module          Yes        The transformer tests are placed inside the valid module.
   Valid test function names  Yes        The transformer tests are named correctly.
   Valid transformer addon    Yes        The addon is configured properly.
   Importable from module     Yes        The transformer can be imported from the parent module.

   In [3]: valid
   Out[3]: True

Unit Tests
""""""""""

* Unit tests should cover specific cases for each of the following methods: ``__init__``,
  ``fit``, ``transform`` and ``reverse_transform``.
* Unit tests for a transformer must have 100% coverage based on the code coverage report.
* The tests should go in a module called ``tests/unit/transformers/{transformer_module}``.

Validating Unit Tests
*********************

The transformer unit tests and their coverage can be validated using the
``validate_transformer_unit_tests`` function. This function returns a ``float`` value representing
the test coverage where 1.0 is 100%. It also prints each test and whether or not it passed. It also
prints a table summarizing the test coverage and provides a link to the full coverage report.

.. code-block:: Python

   In [1]: from tests.contributing import validate_transformer_unit_tests

   In [2]: test_coverage = validate_transformer_unit_tests('rdt.transformers.BooleanTransformer') # Replace BooleanTransformer with your transformer
   Validating source file C:\Datacebo\RDT\rdt\transformers\boolean.py

   ================================================= test session starts =================================================
   collected 12 items

   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test___init__ PASSED                            [  8%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__fit_array PASSED                          [ 16%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__fit_nan_ignore PASSED                     [ 25%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__fit_nan_not_ignore PASSED                 [ 33%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__reverse_transform_2d_ndarray PASSED       [ 41%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__reverse_transform_float_values PASSED     [ 50%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__reverse_transform_float_values_out_of_range PASSED [ 58%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__reverse_transform_nan_ignore PASSED       [ 66%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__reverse_transform_nan_not_ignore PASSED   [ 75%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__reverse_transform_not_null_values PASSED  [ 83%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__transform_array PASSED                    [ 91%]
   tests/unit/transformers/test_boolean.py::TestBooleanTransformer::test__transform_series PASSED                   [100%]

   ============================================ 12 passed, 1 warning in 0.08s ============================================

   SUCCESS: The unit tests passed.
   Name                          Stmts   Miss  Cover   Missing
   -----------------------------------------------------------
   rdt\transformers\boolean.py      37     19    49%   3-36, 40-55, 68, 88, 100
   -----------------------------------------------------------
   TOTAL                            37     19    49%

   ERROR: The unit tests only cover 48.649% of your code.

   Full coverage report here:

   file:///C:/Datacebo/RDT/htmlcov/rdt_transformers_boolean_py.html

   In [3]: test_coverage
   Out [3]: 0.486

Integration Tests
"""""""""""""""""

Integration tests should test the entire workflow of going from input data, to fitting, to
transforming and finally reverse transforming the data. By default, we run integration tests
for each transformer that validate the following checks:

1. The Transformer correctly defines the data type that it supports.
2. At least one Dataset Generator exists for the Transformer data type.
3. The Transformer can transform data and produces outputs of the indicated data types.
4. The Transformer can reverse transform the data it produces, recovering the original data type.
   If ``is_composite_identity``, we expect that the reverse transformed data is equal to the
   original data.
5. The HyperTransformer is able to use the Transformer and produce float values.
6. The HyperTransformer is able to reverse the data that has previously transformed,
   and restore the original data type.

If you wish to test any specific end-to-end scenarios that were not covered in the above checks, 
add a new integration test. Integration tests can be added under
``tests/integration/path/to/test_a_module.py``.

* Before putting up a PR, confirm that the automatic integration tests pass. If new functionality
  that isn't covered is added, feel free to add new integration tests.
* Integration tests should be added under ``tests/unit/transformers/{transformer_module}``.

Validating Integration Tests
****************************

Integration tests can be validated using the ``validate_transformer_integration`` function. This
function returns a boolean representing whether or not the transformer passes all integration
checks. It also prints a table describing each check and whether or not it passed.

.. code-block:: Python

   In [1]: from tests.contributing import validate_transformer_integration

   In [2]: valid = validate_transformer_integration('rdt.transformers.BooleanTransformer') # Replace BooleanTransformer with your transformer
   Validating Integration Tests for transformer BooleanTransformer

   SUCCESS: The integration tests were successful.

   Check                                   Correct    Details
   --------------------------------------  ---------  -----------------------------------------------------------------------------------------------------------------------
   Dataset Generators                      Yes        At least one Dataset Generator exists for the Transformer data type.
   Output Types                            Yes        The Transformer can transform data and produce output(s) of the indicated data type(s).
   Reverse Transform                       Yes        The Transformer can reverse transform the data it produces, going back to the original data type.
   Hypertransformer can transform          Yes        The HyperTransformer is able to use the Transformer and produce float values.
   Hypertransformer can reverse transform  Yes        The HyperTransformer is able to reverse the data that it has previously transformed and restore the original data type.

   In [3]: valid
   Out [3]: True

.. _Transformer Performance:

Transformer Performance
"""""""""""""""""""""""

We want to ensure our transformers are as efficient as possible, in terms of time and memory.
In order to do so, we run performance tests on each transformer, based on the input data type
specified by the transformer.

We generate test data using Dataset Generators. Each transformer should have at least one
Dataset Generator that produces data of the transformer's input type.
If there are any specific dataset characteristics that you think may affect your transformer
performance (e.g. constant data, mostly null data), consider adding a Dataset Generator
for that scenario as well.

.. _Creating Dataset Generators:

Creating Dataset Generators
***************************

In order to test performance, we have a class that is responsible for generating data to test
the transformer methods against. Each subclass implements two static method, ``generate`` 
and ``get_performance_thresholds``.

1. ``generate`` takes in the number of rows to generate, and outputs the expected number
   of data rows.
2. ``get_performance_thresholds`` returns the time and memory threshold for each of the required
   transformer methods. These thresolds are per row.

You should make a generator for every type of column that you believe would be useful to test
against. For some examples, you can look in the `dataset generator folder`_.

The generators each have a ``DATA_TYPE`` class variable. This should match the data type that your
``transformer`` accepts as input.

More details can be found in the `Development Guide`_.

Common Performance Pitfalls
***************************

It is important to keep the performance of these transformers as efficient as possible.
Below are some tips and common pitfalls to avoid when developing your transformer, so as to
optimize performance.

1. Avoid duplicate operations. If you need to do some change to an array/series, try to only
   do it once and reuse that variable later.
2. Try to use vectorized operations when possible.
3. When working with Pandas Series, a lot of the operations are able to handle nulls. If you
   need to round, get the max or get the min of a series, there is no need to filter out nulls
   before doing that calculation.
4. ``pd.to_numeric`` is preferred over ``as_type``.
5. ``pd.to_numeric`` also replaces all None values with NaNs that can be operated on since
   ``np.nan`` is a float type.
6. If you are working with a series that has booleans and null values, there is a
   `nullable boolean type`_ that can be leveraged to avoid having to filter out null values.

Validating Performance
**********************

Validate the performance of your transformer using the ``validate_transformer_performance``
function. This function returns a ``pandas.DataFrame`` containing the performance results
of the transformer.

.. code-block:: Python

   In [1]: from tests.contributing import validate_transformer_performance

   In [2]: results = validate_transformer_performance('rdt.transformers.DatetimeTransformer') # Replace DatetimeTransformer with your transformer
   Validating Performance for transformer DatetimeTransformer

   SUCCESS: The Performance Tests were successful.

   In [3]: results
   Out [3]:
            Evaluation Metric         Value Acceptable     Units  Compared to Average
   0                Fit Memory  9.334700e+01        Yes  Mb / row             0.757455
   1                  Fit Time  6.232677e-07        Yes   s / row             0.574041
   2  Reverse Transform Memory  1.451382e+02        Yes  Mb / row             0.966153
   3    Reverse Transform Time  6.641531e-07        Yes   s / row             1.080660
   4          Transform Memory  8.896317e+01        Yes  Mb / row             0.656664
   5            Transform Time  5.217231e-07        Yes   s / row             0.484631

Fix any performance issues that are reported. If there are no errors but performance
can be improved, this function should be used for reference.

.. _Transformer Quality:

Transformer Quality
"""""""""""""""""""

To assess the quality of a transformer, we run quality tests that apply the Transformer
on all the real world datasets that contain the Transformer input data type. The quality tests
look at how well the original correlations are preserved by using transformed data to train
regression models that predict other columns in the data. We compare the transformer's quality
results to that of other transformers of the same data type.

.. _Adding a Dataset:

Adding a Dataset
****************

If the transformer you are creating adds a new data type, then a dataset with that type may need to
be added for the quality tests. This only needs to be done if the transformer being added is 
expected to preserve or expose relationships in the data. This can be done using the following
steps:

1. Find a dataset containing the data type your transformer uses as an input.

2. Test your transformer against this dataset by loading it into a ``DataFrame`` and using the
   ``get_transformer_regression_scores`` in the ``test_quality`` package::

    from tests.quality.test_quality import get_transformer_regression_scores
    get_transformer_regression_scores(data, data_type, dataset_name, [transformer])

3. If the scores are higher than the ``TEST_THRESHOLD`` in the ``test_quality`` package, contact
   one of the `RDT core contributors`_ on GitHub and ask them to add the dataset. Once this is
   done, the quality tests should pass.

Validating Quality
******************

Validate the quality of your transformer using the ``validate_transformer_quality`` function.
This function returns a ``pandas.DataFrame`` containing the scores attained by the transformer
on each dataset, how that score compares to average and whether or not it is acceptable.

.. code-block:: Python

   In [1]: from tests.contributing import validate_transformer_quality

   In [2]: results = validate_transformer_quality('rdt.transformers.CategoricalTransformer') # Replace CategoricalTransformer with your transformer
   Validating Quality Tests for transformer CategoricalTransformer

   SUCCESS: The quality tests were successful.

   In [3]: results
   Out [3]:
                     Dataset     Score  Compared To Average  Acceptable
   0                   adult  0.223325             0.443181        True
   1      student_placements  0.457490             0.994631        True
   2  student_placements_pii  0.457490             0.988428        True

Fix any quality issues that are reported.

Finalize Your Transformer
"""""""""""""""""""""""""

Re-run all the previous validations until they pass. For a final verification, run
``validate_pull_request`` and fix any errors reported. This function runs all the checks described
above. It also prints a table summarizing the results of all these checks.

.. code-block:: Python

   In [1]: from tests.contributing import validate_pull_request

   In [2]: valid = validate_pull_request('rdt.transformers.BooleanTransformer') # Replace BooleanTransformer with your transformer
   ...................

   Check              Correct    Details
   -----------------  ---------  ----------------------------------------------------------------------
   Code Style         Yes        Code Style is acceptable.
   Unit Tests         Yes        The unit tests are correct and run successfully.
   Integration tests  Yes        The integration tests run successfully.
   Performance Tests  Yes        The performance of the transformer is acceptable.
   Quality tests      Yes        The output data quality is acceptable.
   Clean Repository   Yes        There are no unexpected changes in the repository.

   SUCCESS: The Pull Request can be made!
   You can now commit all your changes, push to GitHub and create a Pull Request.

   In [3]: valid
   Out [3]: True

Once you have done everything above, you can create a PR. Do this by following the steps in the
`Pull Request Guidelines`_ section. Review and fill out the checklist in the PR template to ensure
your code is ready for review.

Summary of Steps to Add a New Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. If it does not exist, open an Issue in Github and describe the Transformer that will be added,
   including the data type that it handles and how it will handle it.
2. Create and clone a fork of the RDT repository.
3. Create a branch in this repository using the naming convention
   issue-[issue-number]-[transformer-name] (eg. issue-123-address-transformer).
4. Implement the Transformer class.
5. Run the ``validate_transformer_code_stye`` function described in the `Code Style`_ section
   and fix the reported errors.
6. Implement Unit Tests for the Transformer.
7. Run the ``validate_transformer_unit_tests`` function and fix the reported errors.
8. Run the ``validate_transformer_integration`` function and fix the reported errors.
9. If required, implement the `Dataset Generators` for the new data type. This is described in the
   `Creating Dataset Generators`_ section.
10. Run the ``validate_transformer_performance`` function and fix any errors reported.
    If there are no errors but performance can be improved, this function should be used for
    reference.
11. If this transformer is expected to help preserve relationships in the data, run the
    ``validate_transformer_quality`` function. If the quality is too low, make the
    necessary enhancements to the transformer.
12. If the quality tests fail because there is no dataset for the transformer's data type,
    follow the steps in the `Adding a Dataset`_ section to add a real world dataset
    containing the new data type to the quality tests.
13. Run the ``validate_pull_request`` function as a final check and fix any errors reported.
14. After all the previous steps pass, all the new and modified files can be committed and pushed
    to github, and a Pull Request can be submitted. Follow the steps in the
    `Pull Request Guidelines`_ section to submit your Pull Request.

.. _Github actions page: https://github.com/sdv-dev/RDT/actions
.. _nullable boolean type: https://pandas.pydata.org/pandas-docs/version/1.0/user_guide/boolean.html
.. _RDT core contributors: https://github.com/orgs/sdv-dev/teams/core-contributors
.. _dataset generator folder: https://github.com/sdv-dev/RDT/tree/master/tests/datasets
.. _Development Guide: DEVELOPMENT.rst