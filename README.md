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

This library is used to apply desired transformations to individual tables or entire datasets
all at once, with the goal of getting completely numeric tables as the output. The desired
transformations can be specified at the column level, or dataset level. For example, you can
apply a datetime transformation to only select columns, or you can specify that you want every
datetime column in the dataset to go through that transformation.

To run the examples, we need to decompress the demo data included in the repository by running this
command on a shell:

```bash
tar -xvzf examples/data/airbnb.tar.gz -C examples/data/
```

## Transforming a column

The base class of this library is the BaseTransformer class. This class provides method to fit
a transformer to your data and transform it, a method to transform new data with an already
fitted transformer and a method to reverse a transform and get data that looks like the original
input. Each transformer class inherits from the BaseTransformer class, and thus has all
these methods.

Transformers take in a column and the meta data for that column as an input. Below we will
demonstrate how to use a datetime transformer to transform and reverse transform a column.

```python
from rdt.transformers import get_col_info
column, column_metadata = get_col_info('users', 'date_account_created', 'examples/data/airbnb/Airbnb_demo_meta.json')
```

The `column` (pandas.DataFrame) variable contains the loaded column information and the `column_metadata` (dict) the field information from the meta.json.


Now we can transform the column.

```python
from rdt.transformers import DTTransformer
transformer = DTTransformer(column_metadata)
transformed_data = transformer.fit_transform(column.to_frame())
```

`transformer_data` (pandas.DataFrame) is the transformed data after fit and transform 
he column with the `DTTransformer`.

If you want to reverse the transformation and get the original data back, you can run the
following command.

```python
transformer.reverse_transform(transformed_data).head(5)
```

And the output will be the reverse transformed data:

```
  date_account_created
0           2014-01-01
1           2014-01-01
2           2014-01-01
3           2014-01-01
4           2014-01-01
```

## Transforming a table

You can also transform an entire table using the HyperTransformer class. Again, we can start by
loading the data.

```python
import json
from rdt.transformers import load_data_table
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
with open(meta_file, "r") as f:
    meta_content = json.loads(f.read())
table_tuple = load_data_table('users', meta_file, meta_content)
table, table_meta = table_tuple
```

Now you can pass a list of the desired transformers into the `fit_transform_table` function to
transform the whole table.

```python
from rdt.hyper_transformer import HyperTransformer
ht = HyperTransformer(meta_file)
transformed = ht.fit_transform_table(table, table_meta, transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer'])
transformed.head(3).T
```

The output will be the transformed table data transposed:

```
                                     0             1             2
?date_account_created     1.000000e+00  1.000000e+00  1.000000e+00
date_account_created      1.388531e+18  1.388531e+18  1.388531e+18
?timestamp_first_active   1.000000e+00  1.000000e+00  1.000000e+00
timestamp_first_active    1.654000e+13  1.654000e+13  1.654000e+13
?date_first_booking       1.000000e+00  0.000000e+00  0.000000e+00
date_first_booking        1.388790e+18  0.000000e+00  0.000000e+00
?gender                   1.000000e+00  1.000000e+00  1.000000e+00
gender                    8.522112e-01  3.412078e-01  1.408864e-01
?age                      1.000000e+00  0.000000e+00  0.000000e+00
age                       6.200000e+01  3.700000e+01  3.700000e+01
?signup_method            1.000000e+00  1.000000e+00  1.000000e+00
signup_method             3.282037e-01  3.500181e-01  4.183867e-01
?signup_flow              1.000000e+00  1.000000e+00  1.000000e+00
signup_flow               4.453093e-01  3.716032e-01  3.906801e-01
?language                 1.000000e+00  1.000000e+00  1.000000e+00
language                  2.927157e-01  5.682538e-01  6.622744e-01
?affiliate_channel        1.000000e+00  1.000000e+00  1.000000e+00
affiliate_channel         9.266169e-01  5.640470e-01  8.044208e-01
?affiliate_provider       1.000000e+00  1.000000e+00  1.000000e+00
affiliate_provider        7.717574e-01  2.539509e-01  7.288847e-01
?first_affiliate_tracked  1.000000e+00  1.000000e+00  1.000000e+00
first_affiliate_tracked   3.861429e-01  8.600605e-01  4.029200e-01
?signup_app               1.000000e+00  1.000000e+00  1.000000e+00
signup_app                6.915504e-01  6.373492e-01  5.798949e-01
?first_device_type        1.000000e+00  1.000000e+00  1.000000e+00
first_device_type         6.271052e-01  2.611754e-01  6.828802e-01
?first_browser            1.000000e+00  1.000000e+00  1.000000e+00
first_browser             2.481743e-01  5.087636e-01  5.023412e-01
```

You can then reverse transform the output to get a table in the original format, but it will
only contain the columns corresponding to those that were transformed (ie. numeric columns).

```python
ht.reverse_transform_table(transformed, table_meta).head(3).T
```

The output will be the reserve transformed table with the original format:

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

The hyper transformer is also capable of transforming all of the tables specified in your
meta.json at once.

```python
from rdt.hyper_transformer import HyperTransformer
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
ht = HyperTransformer(meta_file)
transformed = ht.fit_transform(transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer'])
```

`transformed` (dict) contains all the tables fitted and transformed
 with the given transformers (DDTransformer, NumberTransformer, CatTransformer).

`transformed['tablename']` (pandas.DataFrame) contains the table 'tablename' fitted
 and transfromed data.

And also can revers all the tables transformed:

```python
reverse_transformed = ht.reverse_transform(tables=transformed)
```

Finaly, the `reverse_transformed` (dict) is the reversed transformations
 of the already transformed data.

`reverse_transformed['tablename']` (pandas.DataFrame) contains the table 'tablename'
 reversed data.
