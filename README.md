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

RDT is a Python library used to transform data for data science libraries and preserve
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

In this short tutorial we will guide you to transform columns, tables and datasets.

Before starting, you will need to decompress the demo data included in the repository
by running this command on a shell:

```bash
tar -xvzf examples/data/airbnb.tar.gz -C examples/data/
```

## Transforming a column

### 1. Load column and its metadata

With RDT, transform a column is as simple as specifying the table name,
the column name and the metadata json file.

You can find the metadata format in [SDV documentation](https://hdi-project.github.io/SDV/usage.html#metadata-file-specification).

```python
from rdt.transformers import get_col_info
column, column_metadata = get_col_info('users', 'date_account_created', 'examples/data/airbnb/Airbnb_demo_meta.json')
```

The output is the variable `column` which includes the specified column data:

```
0    2014-01-01
1    2014-01-01
2    2014-01-01
3    2014-01-01
4    2014-01-01
Name: date_account_created, dtype: object
```

And the other output is the variable `column_metadata` which includes the metadata
that corresponds the specified column from the meta:

```json
{'name': 'date_account_created',
 'type': 'datetime',
 'format': '%Y-%m-%d',
 'uniques': 1634}
```

### 2. Load the transformer

In this case in particular, the column correspond to a datetime field,
we will use the 'from rdt.transformers import DTTransformer'.

```python
from rdt.transformers import DTTransformer
transformer = DTTransformer(column_metadata)
transformed_data = transformer.fit_transform(column.to_frame())
```

The output is the variable `transformed_data` which includes
the specified column data transformed:

```
   date_account_created
0          1.388534e+18
1          1.388534e+18
2          1.388534e+18
3          1.388534e+18
4          1.388534e+18
```

### 3. Reverse the transformed data

If you want reverse the transformed data, you can do so by calling
`transformer.reverse_transform` method:

```python
transformer.reverse_transform(transformed_data).head(5)
```

The output includes the specified column data transformed after the reverse:

```
  date_account_created
0           2014-01-01
1           2014-01-01
2           2014-01-01
3           2014-01-01
4           2014-01-01
```

## Transforming a table

### 1. Load table and its metadata

With RDT, transform a table is as simple as specifying the table name,
the metadata json file and the metadata content.

You can find the metadata format in [SDV documentation](https://hdi-project.github.io/SDV/usage.html#metadata-file-specification).

```python
import json
from rdt.transformers import load_data_table
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
with open(meta_file, "r") as f:
    meta_content = json.loads(f.read())
table_tuple = load_data_table('users', meta_file, meta_content)
table, table_meta = table_tuple
```

The output is the variable `table_tuple` which includes a tuple with
the table data and its metadata.

The output of the variable `table` which includes the specified table data:

```
           id date_account_created  ...  first_device_type first_browser
0  d1mm9tcy42           2014-01-01  ...    Windows Desktop        Chrome
1  yo8nz8bqcq           2014-01-01  ...        Mac Desktop       Firefox
2  4grx6yxeby           2014-01-01  ...    Windows Desktop       Firefox
3  ncf87guaf0           2014-01-01  ...    Windows Desktop        Chrome
4  4rvqpxoh3h           2014-01-01  ...             iPhone     -unknown-

[5 rows x 15 columns]
```

And the output of the variable `table_meta` which includes the metadata
that corresponds the specified table the meta:

```json
{'path': 'users_demo.csv',
 'name': 'users',
 'use': True,
 'headers': True,
 'fields': [{'name': 'id',
   'type': 'id',
   'regex': '^.{10}$',
   'uniques': 213451},
   ...
  {'name': 'first_browser',
   'type': 'categorical',
   'subtype': 'categorical',
   'uniques': 52}],
 'primary_key': 'id',
 'number_of_rows': 213451}
```

### 2. Load the transformer

We will be transforming the table using the `from rdt.hyper_transformer import HyperTransformer`.
When transforming a table you need specify the list of transformers to use:

```python
from rdt.hyper_transformer import HyperTransformer
ht = HyperTransformer(meta_file)
transformed = ht.fit_transform_table(table, table_meta, transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer'])
transformed.head(3).T
```

The output is the variable `transformed`, which includes the table transformed:

```
                                     0             1             2
date_account_created      1.388534e+18  1.388534e+18  1.388534e+18
timestamp_first_active    1.388535e+18  1.388535e+18  1.388535e+18
date_first_booking        1.388794e+18  1.388707e+18  1.388707e+18
?date_first_booking       1.000000e+00  0.000000e+00  0.000000e+00
gender                    9.374248e-01  1.726327e-01  1.862825e-01
age                       6.200000e+01  3.700000e+01  3.700000e+01
?age                      1.000000e+00  0.000000e+00  0.000000e+00
signup_method             4.482511e-01  3.434258e-01  2.074619e-01
signup_flow               3.788016e-01  2.725286e-01  4.355109e-01
language                  4.169422e-01  6.413109e-01  4.174422e-01
affiliate_channel         9.252949e-01  3.891049e-01  8.162908e-01
affiliate_provider        7.076034e-01  5.014322e-02  7.414325e-01
first_affiliate_tracked   4.085044e-01  6.479545e-01  3.448558e-01
?first_affiliate_tracked  1.000000e+00  1.000000e+00  1.000000e+00
signup_app                5.505231e-01  6.684141e-01  6.573794e-01
first_device_type         7.263562e-01  3.021326e-01  6.116525e-01
first_browser             1.959058e-01  4.864128e-01  4.935550e-01
```

### 3. Reverse the transformed data

If you want reverse the transformed data, you can do so by calling
`ht.reverse_transform_table` method:

```python
ht.reverse_transform_table(transformed, table_meta).head(3).T
```

The output includes the data transformed after the reverse:

```
                                       0               1                2
date_account_created          2014-01-01      2014-01-01       2014-01-01
timestamp_first_active    19700101053540  19700101053540   19700101053540
date_first_booking            2014-01-04             NaN              NaN
gender                              MALE       -unknown-        -unknown-
age                                   62             NaN              NaN
signup_method                      basic           basic            basic
signup_flow                            0               0                0
language                              en              en               en
affiliate_channel          sem-non-brand          direct        sem-brand
affiliate_provider                google          direct           google
first_affiliate_tracked              omg       untracked              omg
signup_app                           Web             Web              Web
first_device_type        Windows Desktop     Mac Desktop  Windows Desktop
first_browser                     Chrome         Firefox          Firefox
```

## Transforming a dataset

### 1. Load the transformer

With RDT, transform a dataset is as simple as specifying the metadata json file.
Once we have created the instance with the desired metadata we can fit and transforme it.

You can find the metadata format in [SDV documentation](https://hdi-project.github.io/SDV/usage.html#metadata-file-specification).

```python
from rdt.hyper_transformer import HyperTransformer
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
ht = HyperTransformer(meta_file)
transformed = ht.fit_transform(transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer'])
transformed['users']
```

The output of the variable `transformed['users']` which includes the specified table data:

```
                                    0             1             2
date_account_created     1.388534e+18  1.388534e+18  1.388534e+18
timestamp_first_active   1.388535e+18  1.388535e+18  1.388535e+18
date_first_booking       1.388794e+18  1.388707e+18  1.388707e+18
gender                   7.976607e-01  1.903103e-01  3.209892e-01
age                      6.200000e+01  3.700000e+01  3.700000e+01
signup_method            2.655825e-01  5.665843e-01  3.572366e-01
signup_flow              4.169028e-01  3.415770e-01  4.911972e-01
language                 4.790218e-01  6.853290e-01  5.380235e-01
affiliate_channel        9.377552e-01  5.471765e-01  7.957746e-01
affiliate_provider       7.279075e-01  2.566958e-01  7.450474e-01
first_affiliate_tracked  4.225472e-01  7.396975e-01  3.239905e-01
signup_app               4.597071e-01  3.912385e-01  2.820521e-01
first_device_type        6.539733e-01  2.618580e-01  6.375842e-01
first_browser            2.453285e-01  4.943993e-01  5.086062e-01
```

### 2. Reverse the transformed data

If you want reverse the transformed data, you can do so by calling
`ht.reverse_transform` method:

```python
reverse_transformed = ht.reverse_transform(tables=transformed)
reverse_transformed['users']
```

The output of the variable `reverse_transformed['users']` which includes the specified table
data after the reverse:

```
                                       0               1                2
date_account_created          2014-01-01      2014-01-01       2014-01-01
timestamp_first_active    20140101010936  20140101011558   20140101011639
date_first_booking            2014-01-04      2014-01-03       2014-01-03
gender                              MALE       -unknown-        -unknown-
age                                   62              37               37
signup_method                      basic           basic            basic
signup_flow                            0               0                0
language                              en              en               en
affiliate_channel          sem-non-brand          direct        sem-brand
affiliate_provider                google          direct           google
first_affiliate_tracked              omg       untracked              omg
signup_app                           Web             Web              Web
first_device_type        Windows Desktop     Mac Desktop  Windows Desktop
first_browser                     Chrome         Firefox          Firefox
```
