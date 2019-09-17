<p align="left"> 
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“Copulas” />
  <i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi Shield](https://img.shields.io/pypi/v/RDT.svg)](https://pypi.python.org/pypi/RDT)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/RDT.svg?branch=master)](https://travis-ci.org/HDI-Project/RDT)
[![Downloads](https://pepy.tech/badge/rdt)](https://pepy.tech/project/rdt)

# RDT: Reversible Data Transforms

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/RDT

## Overview

RDT is a Python library used to transform data for data science libraries and preserve the transformations in order to reverse them as needed.

# Install

## Requirements

**RDT** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid interfering with other software installed in the system where **RDT** is fun.

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

After creating the virtualenv and activating it, we recommend using [pip](https://pip.pypa.io/en/stable/) in order to install **RDT**:

```bash
pip install rdt
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from sources

Alternatively, with your virtualenv activated, you can clone the repository and install it from source by running `make install` on the `stable` branch:

```bash
git clone https://github.com/HDI-Project/RDT
cd RDT
git checkout stable
make install
```

## Install for development

If you want to contribute to the project, a few more steps are required to make the project ready for development.

First, please head to the GitHub page of the project and make a fork of the project under you own username by clicking on the **fork** button on the upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes the number of issues that you are going to work on:

```bash
git clone https://github.com/HDI-Project/RDT
cd RDT
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, wich will install some additional dependencies for code linting and testing.

```bash
make install
```

Make sure to use them regularly while developing by running the commands `make lint` and `make test`.

# Quickstart

In this guide we will guide you to transform columns, tables and datasets.

To run the examples, we need to decompress the demo data included in the repository by running this
command on a shell:

```bash
tar -xvzf examples/data/airbnb.tar.gz -C examples/data/
```

## Transforming a column

### Load column and its metadata

Transforming a column is simple, you just need import the `get_col_info` method
and call it with the table with the column name to transform and provide a meta.json.

You can find the metadata format in [SDV documentation](https://hdi-project.github.io/SDV/usage.html#metadata-file-specification).

```python
from rdt.transformers import get_col_info
column, column_metadata = get_col_info('users', 'date_account_created', 'examples/data/airbnb/Airbnb_demo_meta.json')
```

The output `column`, which is a `pandas.Series` with the column data:

```
0    2014-01-01
1    2014-01-01
2    2014-01-01
3    2014-01-01
4    2014-01-01
Name: date_account_created, dtype: object
```

And the `column_metadata`, which is a `dict` with the metadata json information
that corresponds to the `date_account_created` column:

```
{'name': 'date_account_created',
 'type': 'datetime',
 'format': '%Y-%m-%d',
 'uniques': 1634}
```

### Load the transformer

The column used in this example correspond to a datetime field,
we will use the `DTTransformer`.

Import the transformer, create a new instance using the metadata,
fit and transform the column:

```python
from rdt.transformers import DTTransformer
transformer = DTTransformer(column_metadata)
transformed_data = transformer.fit_transform(column.to_frame())
```

Note that the `fit_transform` method expect a `pandas.DataFrame`.

The `transformer_data`, which is a `pandas.DataFrame`,
contains the transformed column data:

```
   date_account_created
0          1.388534e+18
1          1.388534e+18
2          1.388534e+18
3          1.388534e+18
4          1.388534e+18
```

### Reverse the transformed data

You can also revese the transformed data using the same transformer:

```python
transformer.reverse_transform(transformed_data).head(5)
```

And the output, which is a `pandas.DataFrame`, will be the reverse transformed data:

```
  date_account_created
0           2014-01-01
1           2014-01-01
2           2014-01-01
3           2014-01-01
4           2014-01-01
```

## Transforming a table

### Load table and its metadata

Transforming a table is similar to transforming a column.

First, you need import the `load_data_table` and load the metadata contend.
Then you need call the method with the table name, the metadata file path
and the metadata file content.

```python
import json
from rdt.transformers import load_data_table
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
with open(meta_file, "r") as f:
    meta_content = json.loads(f.read())
table_tuple = load_data_table('users', meta_file, meta_content)
table, table_meta = table_tuple
```

The `table_tuple`, which is a `tuple`, that contains the table data and it's metadata.
The first value of the tuple is the table data and the second it's metadata.

The `table`, which is a `pandas.DataFrame`, is the table data:

```
           id date_account_created  ...  first_device_type first_browser
0  d1mm9tcy42           2014-01-01  ...    Windows Desktop        Chrome
1  yo8nz8bqcq           2014-01-01  ...        Mac Desktop       Firefox
2  4grx6yxeby           2014-01-01  ...    Windows Desktop       Firefox
3  ncf87guaf0           2014-01-01  ...    Windows Desktop        Chrome
4  4rvqpxoh3h           2014-01-01  ...             iPhone     -unknown-

[5 rows x 15 columns]
```

And the `table_meta`, which is a `dict`, with the metadata json information
that corresponds to the `users` table:

```
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

### Load the transformer

Import the `HyperTransformer` and create a new instance with the metadata filename.

When transforming a table you need specify the list of transformers to use.

```python
from rdt.hyper_transformer import HyperTransformer
ht = HyperTransformer(meta_file)
transformed = ht.fit_transform_table(table, table_meta, transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer'])
transformed.head(3).T
```

The `transformed`, which is a `pandas.DataFrame`, contains the transformed table:

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

### Reverse the transformed data

You can then reverse transform the output to get a table in the original format:

```python
ht.reverse_transform_table(transformed, table_meta).head(3).T
```

The output will be the reserve transformed table, which is a `pandas.DataFrame`,
with the original format:

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

### Load the transformer

The `HyperTransformer` is also capable of transforming all of the tables specified in
your metadata at once, this way you can avoid the load data and it's metadata:

```python
from rdt.hyper_transformer import HyperTransformer
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
ht = HyperTransformer(meta_file)
transformed = ht.fit_transform(transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer'])
```

The `transformed` (dict) contains all the tables fitted and transformed
with the given transformers (DDTransformer, NumberTransformer, CatTransformer).

The transformed dictionary has the next format:

```
{
    'table_name': pandas.DataFrame,
	...
	'table_name': pandas.DataFrame
}
```

The `transformed['table_name']`, which is a `pandas.DataFrame`, contains
the table 'table_name' fitted and transfromed data.

The output below is the `transformed['users']` data:

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

### Reverse the transformed data

And also can revers all the tables transformed at once:

```python
reverse_transformed = ht.reverse_transform(tables=transformed)
```

Finaly, the `reverse_transformed`, which is a `dict`, is the reversed transformations
of the already transformed data.

This dictionary contains the same format that the previous described.

The output below is the `reverse_transformed['users']` data:

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
