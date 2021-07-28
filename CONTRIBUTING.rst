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

Common Performance Pitfalls
"""""""""""""""""""""""""""
It is important to try to keep the performance of these transformers as efficient as possible.
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

Create Performance Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your transformer is complete, please follow the format in this sample `Colab Notebook`_
that runs common performance metrics on the implemented methods and create one for your
transformer. This can help you find inefficiencies in the transformer and may give you ideas
for improvements.

Add Unit and Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There should be unit tests and integration tests created specifically for the new transformer
you add. Unit tests should cover specific cases for each of the following methods: ``__init__``,
``fit``, ``transform`` and ``reverse_transform``. They can be added under
``tests/unit/transformers/{transformer_module}``.

The integration tests should test the whole workflow of going from input data, to fitting, to
transforming and finally reverse transforming the data. The tests should make sure the reversed
data is in the same format or exactly identical to the input data. Integration tests can be
added under ``tests/unit/transformers/{transformer_module}``.

Adding Performance Tests
~~~~~~~~~~~~~~~~~~~~~~~~

Once the new ``transformer`` is complete and has been analyzed for performance, performance tests
should be added for it. If it is a completely new ``transformer``, add a folder with the same name
as it to this directory: ``/tests/performance/test_cases``. Otherwise, add the test cases to the
appropriate existing folder.

The naming convention for the test case files is as follows:

``{description of arguments}_{dataset_generator_name}_{fit_size}_{transform_size}.json``

For example, if we use the default arguments with the ``ConstantIntegerGenerator``, and generate
1000 rows for ``fit`` as well as ``transform``, then we would have the name
``default_ConstantIntegerGenerator_1000_1000.json``.

Each test case configuration file has the following format::

   {
      "dataset": "tests.performance.datasets.UniqueCategories",
      "transformer": "rdt.transformers.categorical.CategoricalTransformer,
      "kwargs": {},
      "fit_rows": 1000,
      "transform_rows": 10000,
      "expected": {
         "fit": {
               "time": 0.3,
               "memory": 400
         },
         "transform": {
               "time": 0.3,
               "memory": 400
         },
         "reverse_transform": {
               "time": 0.3,
               "memory": 400
         }
      }
   }

The configuration should specify the full Python path of the transformer, the keyword arguments
that need to be passed to the transformer when creating its instance, the full Python path of
the dataset generator (explained in more detail below), the number of rows to generate for both
``fit`` and ``transform`` and the max allowable time and memory for each method.

There is a function called ``make_test_case_configs`` in ``tests/performance/test_performance.py``
that can be used to generate test cases once you have the dataset generators created.

Create Dataset generators
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to test performance, we have a class that is responsible for generating data to test
the transformer methods against. Each subclass implements one static method, ``generate`` that
takes in the number of rows to generate. You should make a generator for every type of column
that you believe would be useful to test against. For some examples, you can look in this
folder: https://github.com/sdv-dev/RDT/tree/master/tests/performance/datasets

The generators also each have the following class variables:

1. ``TYPE``
2. ``SUBTYPE``

These should match the type and subtype of data that your ``transformer`` is used for.

Maintainer's Checklist
~~~~~~~~~~~~~~~~~~~~~~

Once you have done everything above, you can create a PR. Be sure to look over the
checklist below to make sure your PR is ready for review.

1. Verify that the profiling notebook was created and used to find any obvious bottlenecks.
2. Verify that performance test cases were created.
3. Verify that the timings and memory values for these test cases are reasonable compared
   to other similar transformers if possible.
4. Verify that unit and integration tests were added for the transformers.
5. Create an issue that is assigned to the user making the PR and verify that the PR resolves
   that issue.
6. Review the ``Pull Request Guidelines`` below.

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
