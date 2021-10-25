.. highlight:: shell

=================
Development Guide
=================

.. contents:: Table of contents
   :local:
   :depth: 3

This guide describes in detail the main technical components of RDT as well as how to develop
them.

RDT Technical Overview
----------------------

The goal of RDT is to be able to transform data that is not machine learning ready into data that
is. By machine learning ready, we mean that the data should consist of data types that most machine
learning models can process. Usually this means outputting numeric data with no nulls. On top of this,
RDT also enforces that those transformations can be reversed, so that data of the original form can
be obtained again. RDT accomplishes this with the use of two main classes:

* ``BaseTransformer``
* ``HyperTransformer``

BaseTransformer
"""""""""""""""

Every Transformer in RDT inherits from the ``BaseTransformer``. The goal of this class or any of
its subclasses is to take data of a certain data type, and convert it into machine learning ready
data. To enable transformers to do this, the ``BaseTransformer`` has the following attributes:

* ``INPUT_TYPE`` (str) - The input type for the transformer.
* ``OUTPUT_TYPES`` (dict) - Dictionary mapping transformed column names to their data types.
* ``DETERMINISTIC_TRANSFORM`` (bool) - Whether or not calling ``transform`` yields a deterministic
  output.
* ``DETERMINISTIC_REVERSE`` (bool) - Whether or not calling ``reverse_transform`` yields a
  deterministic output.
* ``COMPOSITION_IS_IDENTITY`` (bool) - Whether or not calling ``transform`` and then
  ``reverse_transform`` back to back yields data identical to the ``transform`` input (the original
  data).
* ``NEXT_TRANSFORMERS`` (dict) - Dictionary mapping transformed column names to Transformer class names
  to use on them. This is optional. It can also be partially defined, meaning that some transformed
  columns can have an entry in the dict, but not all of them need to.
* ``columns`` (list) - List of column names that the transformer will transform. Set during ``fit``.
* ``output_columns`` (list) - List of column names in the output from calling ``transform``. Set
  during ``fit``.
* ``column_prefix`` (str) - Prefix that will be added to the output columns to make them unique on
  the table. For transformers that act on a single column this is the column name, and for
  transformers that act on multiple columns this is their names joined by `#`. Set during ``fit``.

It also has the following default methods, which the ``Transformer`` subclasses may overwrite when
necessary:

* ``get_output_types()`` - Returns the name of the columns that the transform method creates. By
  default this will be the ``OUTPUT_TYPES`` dictionary with the column names prepended with the
  ``column_prefix`` with a dot (`.`) as the separator.
* ``is_transform_deterministic()`` - Returns a boolean indicating whether the output of the
  ``transform`` method is deterministic. By default this is the value of the
  ``DETERMINISTIC_TRANSFORM`` attribute.
* ``is_reverse_deterministic()`` - Returns a boolean indicating whether the output of the
  ``reverse_transform`` method is deterministic. By default this is the value of the
  ``DETERMINISTIC_REVERSE`` attribute.
* ``is_composition_identity()`` - Returns a boolean indicating whether the output of calling
  ``transform`` and then ``reverse_transform`` on the transformed output is deterministic. By
  default this is the ``COMPOSITION_IS_IDENTITY`` attribute.
* ``get_next_transformers()`` - Returns a dictionary mapping the names of the columns that the
  ``transform`` method creates to the next transformer to use for those columns. By default this
  will be the ``NEXT_TRANSFORMERS`` dictionary with the keys prepended with the ``column_prefix``
  with a dot (`.`) as a separator.
* ``fit(data, columns)`` - Takes in ``pandas.DataFrame`` and list of column names and stores
  the information needed to run ``transform``.
* ``transform(data, drop)`` - Takes in ``pandas.DataFrame`` and bool saying whether or not to
  drop original columns in the output. Returns ``pandas.DataFrame`` containing transformed data.
* ``reverse_transform(data, drop)`` - Takes in ``pandas.DataFrame`` and bool saying whether or
  not to transformed columns in the output. Returns ``pandas.DataFrame`` containing reverse
  transformed data.

Any subclass of the ``BaseTransformer`` can add extra methods or attributes that it needs.

HyperTransformer
""""""""""""""""

The ``HyperTransformer`` class is used to transform an entire table. Under the hood, the
``HyperTransformer`` figures out which Transformer classes to use on each column in order to
get a machine learning ready output. It does so using the following methods:

* ``fit(data)`` - Takes in a ``pandas.DataFrame``. For every column or group of columns in the
  data, it find a transformer to use on it and calls that transformer's `fit` method with those
  columns. If the output of the transformer is not machine learning ready, it will recursively
  find transformers to use on that data type until it is. A sequence of transformers to use is
  created.
* ``transform(data)`` - Takes in a ``pandas.DataFrame``. Goes through the sequence of transformers
  created during ``fit`` and calls their underlying ``transform`` method.
* ``reverse_transform(data)`` - Takes in a ``pandas.DataFrame``. Goes through the sequence of
  transformers created during ``fit`` in reverse and calls their underlying ``reverse_transform``
  method.

Implementing a Transformer
--------------------------

In order to create a new Transformer class, the class should inherit from the ``BaseTransformer``.
It should also set the values for the attributes defined above.

*Note*: Some attributes might not be able to be determined until after ``fit`` is called. In this
case, those attributes should be set in the ``_fit`` method.

The only methods that need to be implemented for a new Transformer class are:

* ``_fit(columns_data)``
* ``_transform(columns_data)``
* ``_reverse_transform()``

Take note of the `_` preceding each method. The ``BaseTransformer`` will call these methods when
``fit``, ``transform`` and ``reverse_transform`` are called. This is because the 
``BaseTransformer`` figures out which columns to pass down behind the scenes. All of the `_`
methods take in a ``pandas.Series`` or ``pandas.DataFrame`` containing only the columns that will
be used by the transformer.

If for some reason, the new transformer requires access to all of the data, then the ``fit``,
``transform`` and ``reverse_transform`` methods can be overwritten.

Example Transformer
"""""""""""""""""""

Now that we have some background information on how Transformers work in RDT, let's create a new
one. For this example, we will create a simple ``USPhoneNumberTransformer``. The goal of this
transformer is to take strings containing phone numbers into numeric data. For the sake of
simplicity, we will assume all phone numbers are of the format `###-###-####` or
`#-###-###-####`.

Let's start by setting the necessary attributes and writing the ``__init__`` method.

.. code-block:: Python

    class USPhoneNumberTransformer(BaseTransformer):

        INPUT_TYPE = 'phone_number'
        DETERMINISTIC_TRANSFORM = True
        DETERMINISTIC_REVERSE = True
        COMPOSITION_IS_IDENTITY = True

        def __init__(self):
            self.has_country_code = None

Now we can write the ``_fit`` method.

.. code-block:: Python

    def _fit(self, columns_data):
        number = ''.join(columns_data.loc[0].split('-'))
        self.has_country_code = len(number) == 11

Since the ``country_code`` may or may not be present, we can overwrite the
``get_next_transformers`` and ``get_output_types`` methods accordingly.

.. code-block:: Python

    def get_output_types(self):
        output_types = {
            'area_code': 'categorical',
            'exchange': 'integer',
            'line': 'integer'
        }
        if self.has_country_code:
            output_types['country_code'] = 'categorical'

        return self._add_prefix(output_types)

    def get_next_transformers(self):
        next_transformers = {
            'country_code': 'CategoricalTransformer',
            'area_code': 'CategoricalTransformer'
        }
        if self.has_country_code:
            next_transformers['country_code'] = 'CategoricalTransformer'
        
        return self._add_prefix(next_transformers)

``_add_prefix`` is a private method that prepends the ``column_prefix`` attributes to every key
in a dictionary. Now that we have this information, we can write the ``_transform`` and
``_reverse_transform`` methods.

.. code-block:: Python

    def _transform(self, data):
        return data.str.split('-', expand=True)

    def _reverse_transform(self, data):
        if self.has_country_code:
            country_code = data.iloc[:, 0].astype('str')
            area_code = data.iloc[:, 1].astype('str')
            exchange = data.iloc[:, 2].astype('str')
            line = data.iloc[:, 3].astype('str')
            return country_code + '-' + area_code + '-' + exchange + '-' + line
        
        area_code = data.iloc[:, 0].astype('str')
        exchange = data.iloc[:, 1].astype('str')
        line = data.iloc[:, 2].astype('str')
        return area_code + '-' + exchange + '-' + line

We don't have to worry about the naming of the output columns because the ``BaseTransformer``
handles that for us. Let's view the complete class below.

.. code-block:: Python
    class USPhoneNumberTransformer(BaseTransformer):

        INPUT_TYPE = 'phone_number'
        DETERMINISTIC_TRANSFORM = True
        DETERMINISTIC_REVERSE = True
        COMPOSITION_IS_IDENTITY = True

        def __init__(self):
            self.has_country_code = None
        
        def _fit(self, columns_data):
            number = ''.join(columns_data.loc[0].split('-'))
            self.has_country_code = len(number) == 11

        def get_output_types(self):
            output_types = {
                'area_code': 'categorical',
                'exchange': 'integer',
                'line': 'integer'
            }
            if self.has_country_code:
                output_types['country_code'] = 'categorical'

            return self._add_prefix(output_types)

        def get_next_transformers(self):
            next_transformers = {
                'country_code': 'CategoricalTransformer',
                'area_code': 'CategoricalTransformer'
            }
            if self.has_country_code:
                next_transformers['country_code'] = 'CategoricalTransformer'

            return self._add_prefix(next_transformers)
        
        def _transform(self, data):
            return data.str.split('-', expand=True)

        def _reverse_transform(self, data):
            if self.has_country_code:
                country_code = data.iloc[:, 0].astype('str')
                area_code = data.iloc[:, 1].astype('str')
                exchange = data.iloc[:, 2].astype('str')
                line = data.iloc[:, 3].astype('str')
                return country_code + '-' + area_code + '-' + exchange + '-' + line
            
            area_code = data.iloc[:, 0].astype('str')
            exchange = data.iloc[:, 1].astype('str')
            line = data.iloc[:, 2].astype('str')
            return area_code + '-' + exchange + '-' + line

Now we can see our `USPhoneNumberTransformer` in action.

.. code-block:: Python

    transformer = USPhoneNumberTransformer()
    data = pd.DataFrame({
        'phone_numbers': ['1-773-404-7845', '1-773-543-4780', '1-111-111-1111']
    })
    transformer.fit(data, ['phone_numbers'])
    transformed = transformer.transform(data)
    print(transformed)
    reversed = transformer.reverse_transform(transformed)
    print(reversed)

We can also run it using the `HyperTransformer`.

.. code-block:: Python

    ht = HyperTransformer(
        data_type_transformers={'phone_number': USPhoneNumberTransformer},
        field_types={'phone_numbers': 'phone_number'}
    )
    ht.fit(data)
    transformed = ht.transform(data)
    print(transformed)
    reversed = ht.reverse_transform(transformed)
    print(reversed)

Dataset Generators
------------------

In RDT, performance tests are run to assure that each transformer is efficient. In order to run
these tests, we have classes that generate datasets of a certain data type. If a new transformer
introduces a new data type, the a ``DatasetGenerator`` class will need to be added for it.

BaseDatasetGenerator
""""""""""""""""""""

All dataset generators inherit from the ``BaseDatasetGenerator`` class. It has the following
class attribute:

* ``DATA_TYPE`` (str) - The data type for the class to generate.

They must implement the following methods.

* ``generate(num_rows)`` - Takes in an int representing the number of rows to generate. Returns a
  ``numpy.ndarray`` of size ``num_rows`` where each value is of the class' ``DATA_TYPE``.

* ``get_performance_thresholds()`` - Returns a dict mapping each of the main methods for a
  transformer (``fit``, ``transform``, ``reverse_transform``) to the expected time and memory it
  takes for those methods to run on 1 row.

Implementing a DatasetGenerator
"""""""""""""""""""""""""""""""

To create a new ``DatasetGenerator``, the methods described above need to be implemented. The
class should be placed in a new file in the following location ``tests/datasets/{DATA_TYPE}.py``.
Each generator must inherit from the base class as well as ``abc.ABC``.

Example DatasetGenerator
************************

Let's create a ``DatasetGenerator`` for the ``phone_number`` data type that we introduced earlier.
We can start by implementing the ``generate`` method and setting the ``DATA_TYPE``.

.. code-block:: Python

    from abc import ABC

    import numpy as np

    from tests.datasets.base import BaseDatasetGenerator

    class USPhoneNumberGenerator(BaseDatasetGenerator, ABC):
        DATA_TYPE = 'phone_number'
        
        @staticmethod
        def generate(num_rows):
            area_codes = np.random.randint(low=100, high=999, size=num_rows).astype(str)
            exchange = np.random.randint(low=100, high=999, size=num_rows).astype(str)
            line = np.random.randint(low=1000, high=9999, size=num_rows).astype(str)
            return np.apply_along_axis('-'.join, 0, [area_codes, exchange, line])

In order for the tests to run, the generator must also implement the ``get_performance_thresholds``
method. The times are specified in seconds and the memory in bytes.

.. code-block:: Python

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            'fit': {
                'time': 1,
                'memory': 100.0
            },
            'transform': {
                'time': 1,
                'memory': 1000.0
            },
            'reverse_transform': {
                'time': 1,
                'memory': 1000.0,
            }
        }

To view the result of the generator we can run the following:

.. code-block:: Python

    USPhoneNumberGenerator.generate(100)
