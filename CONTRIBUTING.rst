.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

Getting Started!
----------------

Ready to contribute? Here's how to set up `Reversible Data Transforms` for local development.

1. Fork the `Reversible Data Transforms` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/RDT.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv RDT
    $ cd RDT/
    $ make install-develop

4. Claim or file an issue on GitHub::

   If there is already an issue on GitHub for the contribution you wish to make, claim it.
   If not, please file an issue and then claim it before creating a branch.

5. Create a branch for local development::

    $ git checkout -b issue-[issue-number]-description-of-your-bugfix-or-feature

   The naming scheme for your branch should have a prefix of the format ``issue-X``
   where X is the associated issue number, such as ``issue-3-fix-foo-bug``. If you
   are not developing on your own fork, further prefix the branch with your GitHub
   username, like ``githubusername/gh-3-fix-foo-bug``.

   Now you can make your changes locally.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-branch

Adding a New Transformer
------------------------

Create Transformer Class
~~~~~~~~~~~~~~~~~~~~~~~~

All new Transformer classes should inherit from `BaseTransformer` or one of its child classes.
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

Unit Tests
~~~~~~~~~~
Unit tests should cover specific cases for each of the following methods: ``__init__``,
``fit``, ``transform`` and ``reverse_transform``.

Creating Unit Tests
"""""""""""""""""""

There should be unit tests created specifically for the new transformer you add.
The unit tests are expected to cover 100% of your transformer's code based on the
coverage report. All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests should go in a module called ``tests/unit/transformers/{transformer_module}``.
   Note that the module name has the ``test_`` prefix and is located in a path similar
   to the one of the tested module, just inside the ``tests`` folder.

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

Tests can be run locally using::
    $ python -m pytest tests.test_rdt

Specific tests can be singled out using::
    $ python -m pytest -k 'foo'

Integration Tests
~~~~~~~~~~~~~~~~~

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

Add Integration Tests
"""""""""""""""""""""

If you wish to test any specific end-to-end scenarios that were not covered in the above checks, 
add a new integration test. Integration tests can be added under
``tests/unit/transformers/{transformer_module}``.

Transformer Performance
~~~~~~~~~~~~~~~~~~~~~~~

We want to ensure our transformers are as efficient as possible, in terms of time and memory.
In order to do so, we run performance tests on each transformer, based on the input data type
specified by the transformer.

We generate test data using Dataset Generators. Each transformer should have at least one
Dataset Generator that produces data of the transformer's input type.
If there are any specific dataset characteristics that you think may affect your transformer
performance (e.g. constant data, mostly null data), consider adding a Dataset Generator
for that scenario as well.

Creating Dataset Generators
"""""""""""""""""""""""""""

In order to test performance, we have a class that is responsible for generating data to test
the transformer methods against. Each subclass implements two static method, ``generate`` 
and ``get_performance_thresholds``.

1. ``generate`` takes in the number of rows to generate, and outputs the expected number
   of data rows.
2. ``get_performance_thresholds`` returns the time and memory threshold for each of the required
   transformer methods. These thresolds are per row.

You should make a generator for every type of column that you believe would be useful to test
against. For some examples, you can look in this
folder: https://github.com/sdv-dev/RDT/tree/master/tests/datasets

The generators each have a ``DATA_TYPE`` class variable. This should match the data type that your
``transformer`` accepts as input.

Common Performance Pitfalls
"""""""""""""""""""""""""""

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

Transformer Quality
~~~~~~~~~~~~~~~~~~~

To assess the quality of a transformer, we run quality tests that apply the Transformer
on all the real world datasets that contain the Transformer input data type. The quality tests
look at how well the original correlations are preserved by using transformed data to train
regression models that predict other columns in the data. We compare the transformer's quality
results to that of other transformers of the same data type.

Adding a Dataset
""""""""""""""""

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

Validate Code
~~~~~~~~~~~~~

The following functions should be run to assure that your code is ready for a pull request.

Validate Code Style
~~~~~~~~~~~~~~~~~~~

Validate the style of your code using the ``validate_transformer_code_style`` function::

    from tests.contribution import validate_transformer_code_style
    validate_transformer_code_style('rdt.transformers.<YourTransformer>')

Fix any style errors that are reported.

Validate Unit Tests
"""""""""""""""""""

Validate the results and coverage of your transformer's unit tests using the
``validate_transformer_unit_tests`` function::

    from tests.contribution import validate_transformer_unit_tests
    validate_transformer_unit_tests('rdt.transformers.<YourTransformer>')

Fix any unit test errors that are reported.

Validate Integration Tests
""""""""""""""""""""""""""

Validate the results of your transformer's integration tests using the
``validate_transformer_integration`` function::

    from tests.contribution import validate_transformer_integration
    validate_transformer_integration('rdt.transformers.<YourTransformer>')

Fix any integration test errors that are reported.

Validate Performance
""""""""""""""""""""

Validate the performance of your transformer using the
``validate_transformer_performance`` function::

    from tests.contribution import validate_transformer_performance
    validate_transformer_performance('rdt.transformers.<YourTransformer>')

Fix any performance issues that are reported. If there are no errors but performance
can be improved, this function should be used for reference.

Validate Quality
""""""""""""""""

Validate the quality of your transformer using the
``validate_transformer_quality`` function::

    from tests.contribution import validate_transformer_quality
    validate_transformer_quality('rdt.transformers.<YourTransformer>')

Fix any quality issues that are reported. If there are no errors but quality can be improved,
this function should be used for reference.

If there are no results, this means that we do not have a real world dataset with your
transformer's data type. Please follow the steps in the ``Adding a Dataset`` section if
this happens.

Finalize Your Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~

Re-run all the previous validations until they pass. For a final verification, run
``validate_pull_request`` and fix any errors reported::

    from tests.contribution import validate_pull_request
    validate_pull_request('rdt.transformers.<YourTransformer>')

Once you have done everything above, you can create a PR. Follow the steps below to create a PR.
Review and fill out the checklist in the PR template to ensure your code is ready for review.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code
4. The pull request should work for all the supported Python versions. Check the `Github actions
   page`_ and make sure that all the checks pass.

.. _nullable boolean type: https://pandas.pydata.org/pandas-docs/version/1.0/user_guide/boolean.html
.._RDT core contributors: https://github.com/orgs/sdv-dev/teams/core-contributors
