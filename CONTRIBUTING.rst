.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at the `GitHub Issues page`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Reversible Data Transforms could always use more documentation, whether as part of the
official Reversible Data Transforms docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at the `GitHub Issues page`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Adding a New Transformer
------------------------

Create Transformer Class
~~~~~~~~~~~~~~~~~~~~~~~~

All new Transformer classes should inherit from `BaseTransformer` or one of its child classes.
There are only three required methods for a transformer:

1. ``fit``: Used to store and learn any values from the input data that might be useful
   for the transformer.
2. ``transform``: Used to transform the input data into completely numeric data. This method
   should not modify the internal state of the Transformer instance.
3. ``reverse_transform``: Used to convert data that is completely numeric back into the
   format of the fitted data. This method should not modify the internal state of the
   Transformer instance.

Each transformer class should be placed inside the ``rdt/transformers`` folder, in a module
file named after the data type that the transformer operates on. For example, if you are
writing a transformer that works with ``categorical`` data, your transformer should be placed
inside the ``rdt/transformers/categorical.py`` module.

Validate Code Style
~~~~~~~~~~~~~~~~~~~

Validate the style of your code using the ``validate_transformer_code_style`` function::

    from tests.contribution import validate_transformer_code_style
    validate_transformer_code_style('rdt.transformers.<YourTransformer>')

Fix any style errors that are reported.

Unit Tests
~~~~~~~~~~
Unit tests should cover specific cases for each of the following methods: ``__init__``,
``fit``, ``transform`` and ``reverse_transform``.

Creating Unit Tests
"""""""""""""""""""

There should be unit tests created specifically for the new transformer you add.
They can be added under ``tests/unit/transformers/{transformer_module}``. The unit tests are
expected to cover 100% of your transformer's code.

Validate Unit Tests
"""""""""""""""""""

Validate the results and coverage of your transformer's unit tests using the
``validate_transformer_unit_tests`` function::

    from tests.contribution import validate_transformer_unit_tests
    validate_transformer_unit_tests('rdt.transformers.<YourTransformer>')

Fix any unit test errors that are reported.

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

Validate Integration Tests
""""""""""""""""""""""""""

Validate the results of your transformer's integration tests using the
``validate_transformer_integration`` function::

    from tests.contribution import validate_transformer_integration
    validate_transformer_integration('rdt.transformers.<YourTransformer>')

Fix any integration test errors that are reported.

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

The generators each have a ``DATA_TYPE`` class variable. This should match the data type that your ``transformer`` accepts as input.

Validate Performance
""""""""""""""""""""

Validate the performance of your transformer using the
``validate_transformer_performance`` function::

    from tests.contribution import validate_transformer_performance
    validate_transformer_performance('rdt.transformers.<YourTransformer>')

Fix any performance issues that are reported. If there are no errors but performance
can be improved, this function should be used for reference.

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

To assess the quality of a transformer, We run quality tests that apply the Transformer
n all the real world datasets that contain the Transformer input data type. The quality tests
look at how well the original correlations are preserved and how good a synthetic data generator
is when trained on the data produced by this Transformer. We compare the transformer's quality
results to that of other transformers of the same Data Type.

Validate Quality
""""""""""""""""

Validate the quality of your transformer using the
``validate_transformer_quality`` function::

    from tests.contribution import validate_transformer_quality
    validate_transformer_quality('rdt.transformers.<YourTransformer>')

Fix any quality issues that are reported. If there are no errors but quality can be improved,
this function should be used for reference.

If there are no results, this means that we do not have a real world dataset with your
transformer's data type. Please find a suitable dataset and open an issue requesting for it
to be added.

Finalize Your Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~

Re-run all the previous validations until they pass. For a final verification, run
``validate_pull_request`` and fix any errors reported::

    from tests.contribution import validate_pull_request
    validate_pull_request('rdt.transformers.<YourTransformer>')

Once you have done everything above, you can create a PR. Follow the steps below to create a PR.
Review and fill out the checklist in the PR template to ensure your code is ready for review.

Get Started!
------------

Ready to contribute? Here's how to set up `Reversible Data Transforms` for local development.

1. Fork the `Reversible Data Transforms` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/RDT.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv RDT
    $ cd RDT/
    $ make install-develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Try to use the naming scheme of prefixing your branch with ``gh-X`` where X is
   the associated issue, such as ``gh-3-fix-foo-bug``. And if you are not
   developing on your own fork, further prefix the branch with your GitHub
   username, like ``githubusername/gh-3-fix-foo-bug``.

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   For this, make sure to run the tests suite and check the code coverage::

    $ make lint       # Check code styling
    $ make test       # Run the tests
    $ make coverage   # Get the coverage report

6. When you're done making changes, check that your changes pass all the styling checks and
   tests, including other Python supported versions, using::

    $ make test-all

7. Make also sure to include the necessary documentation in the code as docstrings following
   the `Google docstrings style`_.
   If you want to view how your documentation will look like when it is published, you can
   generate and view the docs with this command::

    $ make view-docs

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the documentation in an appropriate place.
5. The pull request should work for all the supported Python versions. Check the `Github actions
   page`_ and make sure that all the checks pass.

Unit Testing Guidelines
-----------------------

All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests that cover a module called ``rdt/path/to/a_module.py``
   should be implemented in a separated module called
   ``tests/rdt/path/to/test_a_module.py``.
   Note that the module name has the ``test_`` prefix and is located in a path similar
   to the one of the tested module, just inside the ``tests`` folder.

3. Each method of the tested module should have at least one associated test method, and
   each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have descriptive names
   that indicate which scenario they cover.
   Names such as ``test_some_methed_input_none``, ``test_some_method_value_error`` or
   ``test_some_method_timeout`` are right, but names like ``test_some_method_1``,
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

Tips
----

To run a subset of tests::

    $ python -m pytest tests.test_rdt
    $ python -m pytest -k 'foo'

Release Workflow
----------------

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in ``setup.cfg``, ``rdt/__init__.py`` and
   ``HISTORY.md`` files.
3. Create a new git tag pointing at the corresponding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
5. Update the version in ``setup.cfg`` and ``rdt/__init__.py``
   to open the next development iteration.

.. note:: Before starting the process, make sure that ``HISTORY.md`` has been updated with a new
          entry that explains the changes that will be included in the new version.
          Normally this is just a list of the Pull Requests that have been merged to master
          since the last release.

Once this is done, run of the following commands:

1. If you are releasing a patch version::

    make release

2. If you are releasing a minor version::

    make release-minor

3. If you are releasing a major version::

    make release-major

Release Candidates
~~~~~~~~~~~~~~~~~~

Sometimes it is necessary or convenient to upload a release candidate to PyPi as a pre-release,
in order to make some of the new features available for testing on other projects before they
are included in an actual full-blown release.

In order to perform such an action, you can execute::

    make release-candidate

This will perform the following actions:

1. Build and upload the current version to PyPi as a pre-release, with the format ``X.Y.Z.devN``

2. Bump the current version to the next release candidate, ``X.Y.Z.dev(N+1)``

After this is done, the new pre-release can be installed by including the ``dev`` section in the
dependency specification, either in ``setup.py``::

    install_requires = [
        ...
        'rdt>=X.Y.Z.dev',
        ...
    ]

or in command line::

    pip install 'rdt>=X.Y.Z.dev'


.. _GitHub issues page: https://github.com/sdv-dev/RDT/issues
.. _Github actions page: https://github.com/sdv-dev/RDT/actions
.. _Google docstrings style: https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments
.. _nullable boolean type: https://pandas.pydata.org/pandas-docs/version/1.0/user_guide/boolean.html
.. _Colab Notebook: https://colab.research.google.com/drive/1dGnBLMW-5LATGoBUuQKWfOTZFssBmgYu?usp=sharing
