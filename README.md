<p align="left"> 
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“Copulas” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi Shield](https://img.shields.io/pypi/v/RDT.svg)](https://pypi.python.org/pypi/RDT)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/RDT.svg?branch=master)](https://travis-ci.org/HDI-Project/RDT)
[![Downloads](https://pepy.tech/badge/rdt)](https://pepy.tech/project/rdt)

# RDT: Reversible Data Transforms

- License: MIT
- Documentation: https://HDI-Project.github.io/RDT
- Homepage: https://github.com/HDI-Project/RDT

## Overview

**RDT** is a Python library used to transform data for data science libraries and preserve
the transformations in order to reverse them as needed.

# Install

## Requirements

**RDT** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **RDT** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **RDT**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) rdt-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source rdt-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **RDT**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **RDT**:

```bash
pip install rdt
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from sources

Alternatively, with your virtualenv activated, you can clone the repository and install
it from source by running `make install` on the `stable` branch:

```bash
git clone https://github.com/HDI-Project/RDT
cd RDT
git checkout stable
make install
```

For development, you can use `make install-develop` instead in order to install all
the required dependencies for testing and code linting.

# Quickstart

In this short series of tutorials we will guide you through a series of steps that will
help you getting started using **RDT** to transform columns, tables and datasets.

## Preparing the demo data

Before starting, you will need to decompress the demo data included in the repository
by running this command on a shell:

```bash
tar -xvzf examples/data/airbnb.tar.gz -C examples/data/
```

## Transforming a column

In this first guide, you will learn how to use **RDT** in its simplest form, transforming
a single column loaded as a `pandas.DataFrame` object.

### 1. Load the column and its metadata

In order to load a column and its metadata, you must call the `rdt.load_data` function passing
it the path to the metadata json file, the name of the table from which to load the column,
and the name of the column to load.

You can find documentation about the metadata format in [MetaData.json](
https://github.com/HDI-Project/MetaData.json).

```python
from rdt import load_data

metadata_path = 'tests/data/airbnb/airbnb_meta.json'

column_data, column_metadata = load_data(
    metadata_path=metadata_path,
    table_name='users',
    column_name='date_account_created',
)
```

The output will be the variable `column_data`, which is a `pandas.DataFrame` with the column data:

```
  date_account_created
0           2014-01-01
1           2014-01-01
2           2014-01-01
3           2014-01-01
4           2014-01-01
```

And the `column_metadata`, which is a `dict` containing the information from the metadata json
that corresponds to this column:

```
{
    'name': 'date_account_created',
    'type': 'datetime',
    'format': '%Y-%m-%d',
    'uniques': 1634
}
```

### 2. Load the transformer

In this case the column is a datetime, so we will use the `DTTransformer`.

```python
from rdt.transformers import DTTransformer
transformer = DTTransformer(column_metadata)
```

In order to transform the data, we will call its `fit_transform` method passing the
`column` data:

```python
transformed_data = transformer.fit_transform(column_data)
```

The output will be another `pandas.DataFrame` with the transformed data:

```
   date_account_created
0          1.388534e+18
1          1.388534e+18
2          1.388534e+18
3          1.388534e+18
4          1.388534e+18
```

### 3. Reverse the transformed data

In order to reverse the previous transformation, the transformed data can be passed to
the `reverse_transform` method of the transformer:

```python
reversed_data = transformer.reverse_transform(transformed_data)
```

The output will be a `pandas.DataFrame` containing the data from which the transformed data
was generated with.

In this case, of course, the obtained data should be identical to the original one:

```
  date_account_created
0           2014-01-01
1           2014-01-01
2           2014-01-01
3           2014-01-01
4           2014-01-01
```

## Transforming a table

Once we know how to transform a single column, we can try to go the next level and transform
a table with multiple columns.

### 1. Load the table data and its metadata

In order to load a complete table, we will use the same `rdt.load_data` function as before,
but omit the `column_name` from the call.

```python
table_data, table_metadata = load_data(
    metadata_path=metadata_path,
    table_name='users',
)
```

The output, like before will be compsed by the `table_data`, which in this case will contain
all the columns from the table:

```
           id date_account_created  timestamp_first_active  ... signup_app first_device_type  first_browser
0  d1mm9tcy42           2014-01-01          20140101000936  ...        Web   Windows Desktop         Chrome
1  yo8nz8bqcq           2014-01-01          20140101001558  ...        Web       Mac Desktop        Firefox
2  4grx6yxeby           2014-01-01          20140101001639  ...        Web   Windows Desktop        Firefox
3  ncf87guaf0           2014-01-01          20140101002146  ...        Web   Windows Desktop         Chrome
4  4rvqpxoh3h           2014-01-01          20140101002619  ...        iOS            iPhone      -unknown-
```

And the `table_metadata`, which will also contain all the information available about the table:

```
{
    'path': 'users_demo.csv',
    'name': 'users',
    'use': True,
    'headers': True,
    'fields': [
        {
            'name': 'id',
            'type': 'id',
            'regex': '^.{10}$',
            'uniques': 213451
        },
        ...
        {
            'name': 'first_browser',
            'type': 'categorical',
            'subtype': 'categorical',
            'uniques': 52
        }
    ],
    'primary_key': 'id',
    'number_of_rows': 213451
}
```

### 2. Load the transformer

In order to manuipulate a complete table we will need to import the `rdt.HyperTransformer` class
and create an instance of it passing it the path to our metadata file.

```python
from rdt import HyperTransformer
ht = HyperTransformer(metadata=metadata_path)
```

### 3. Transform the table data

In order to transform the data, we will call the `fit_transform_table` method from our
`HyperTransformer` instance passing it the table data, the table metadata and the names of the
transformers that we want to apply.

```python
transformed = ht.fit_transform_table(
    table=table_data,
    table_meta=table_metadata,
    transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer']
)
```

The output, again, will be the transformed data:

```
         id  date_account_created  timestamp_first_active  ...  signup_app  first_device_type  first_browser
0  0.512195          1.388534e+18            1.388535e+18  ...    0.204759           0.417261       0.423842
1  0.958701          1.388534e+18            1.388535e+18  ...    0.569893           0.115335       0.756304
2  0.106468          1.388534e+18            1.388535e+18  ...    0.381164           0.571280       0.869942
3  0.724346          1.388534e+18            1.388536e+18  ...    0.485542           0.668070       0.364122
4  0.345691          1.388534e+18            1.388536e+18  ...    0.944064           0.847751       0.108216
```

### 4. Reverse the transformation

In order to reverse the transformation and recover the original data from the transformed one,
we need to call `reverse_transform_table` of the `HyperTransformer` instance passing it the
transformed data and the table metadata.

```python
reversed_data = ht.reverse_transform_table(
    table=transformed,
    table_meta=table_metadata
)
```

The output will be the reversed data. Just like before, this should look exactly like the
original data:

```
           id date_account_created timestamp_first_active  ... signup_app first_device_type  first_browser
0  d1mm9tcy42           2014-01-01         20140101010936  ...        Web   Windows Desktop         Chrome
1  yo8nz8bqcq           2014-01-01         20140101011558  ...        Web       Mac Desktop        Firefox
2  4grx6yxeby           2014-01-01         20140101011639  ...        Web   Windows Desktop        Firefox
3  ncf87guaf0           2014-01-01         20140101012146  ...        Web   Windows Desktop         Chrome
4  4rvqpxoh3h           2014-01-01         20140101012619  ...        iOS            iPhone      -unknown-
```
