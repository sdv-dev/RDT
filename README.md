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

In order to load the column and its metadata, you must call the `get_col_info` function passing it
the table from which to load the column, the name of the column to load, and the path to the
metadata json file.

You can find the metadata format in [MetaData.json](https://github.com/HDI-Project/MetaData.json).

```python
from rdt.transformers import get_col_info

column, column_metadata = get_col_info(
    table_name='users',
    col_name='date_account_created',
    meta_file='examples/data/airbnb/Airbnb_demo_meta.json'
)
```

The output is the variable `column`, which is a `pandas.Series` with the column data:

```
0    2014-01-01
1    2014-01-01
2    2014-01-01
3    2014-01-01
4    2014-01-01
Name: date_account_created, dtype: object
```

The output is the variable `column_metadata`, which is a `dict` with the column metadata:

```
{
    'name': 'date_account_created',
    'type': 'datetime',
    'format': '%Y-%m-%d',
    'uniques': 1634
}
```

### 2. Load the transformer

In this case, since the column is a datetime, we will use the DTTransformer.

```python
from rdt.transformers import DTTransformer
transformer = DTTransformer(column_metadata)
```

And call its `fit_transform` method passing the `column` data as a DataFrame:

```python
transformed_data = transformer.fit_transform(column.to_frame())
```

The output will be the transformed data:

```
   date_account_created
0          1.388534e+18
1          1.388534e+18
2          1.388534e+18
3          1.388534e+18
4          1.388534e+18
```

### 3. Reverse the transformed data

In order to reverse the transformed data call `reverse_transform` of the transformer
with the transformed data.

```python
transformer.reverse_transform(transformed_data).head(5)
```

The output will be the transformed data reversed:

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

In order to load a table and its metadata, you must call the `load_data_table` function passing it
the table name, the path to the metadata json file and the metadata content as a dict.

You can find the metadata format in [MetaData.json](https://github.com/HDI-Project/MetaData.json).

```python
import json
from rdt.transformers import load_data_table

meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'

with open(meta_file, "r") as f:
    meta_content = json.loads(f.read())

table_tuple = load_data_table(
    table_name='users',
    meta_file=meta_file,
    meta=meta_content
)

table, table_meta = table_tuple
```

The variable `table_tuple`, which is a `tuple` with the table data and its metadata.

The output is the variable `table`, which is a `pandas.DataFrame` with the table data:

```
           id date_account_created  ...  first_device_type first_browser
0  d1mm9tcy42           2014-01-01  ...    Windows Desktop        Chrome
1  yo8nz8bqcq           2014-01-01  ...        Mac Desktop       Firefox
2  4grx6yxeby           2014-01-01  ...    Windows Desktop       Firefox
3  ncf87guaf0           2014-01-01  ...    Windows Desktop        Chrome
4  4rvqpxoh3h           2014-01-01  ...             iPhone     -unknown-

[5 rows x 15 columns]
```

The output is the variable `table_meta`, which is a `dict` with the table metadata:

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

In this case, since we want transform a table, we will use the Hypertransformer passing
the metadata json file.

```python
from rdt.hyper_transformer import HyperTransformer
ht = HyperTransformer(metadata=meta_file)
```

And call its `fit_transform_table` method passing the table data, the table metadata and
`transformer_list` as a list of strings with the transformers to be used.

```python
transformed = ht.fit_transform_table(
    table=table,
    table_meta=table_meta,
    transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer']
)
transformed.head(3).T
```

The output will be the transformed data:

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

In order to reverse the transformed data call `reverse_transform_table` of the transformer
with the transformed data.

```python
ht.reverse_transform_table(transformed, table_meta).head(3).T
```

The output will be the transformed data reversed:

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

In order to load the dataset, you must instance a new HyperTransformer passing
the metadata json file path.

You can find the metadata format in [MetaData.json](https://github.com/HDI-Project/MetaData.json).

```python
from rdt.hyper_transformer import HyperTransformer
meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
ht = HyperTransformer(metadata=meta_file)
```

And call its `fit_transform_table` method passing the `transformer_list` as a list of strings
with the transformers to be used.

```python
transformed = ht.fit_transform(
    transformer_list=['DTTransformer', 'NumberTransformer', 'CatTransformer']
)
```

The output will be the transformed data:

```
{
    'users': date_account_created  timestamp_first_active  ...  first_device_type  first_browser
        0            1.388534e+18            1.388535e+18  ...           0.688490       0.310197
        1            1.388534e+18            1.388535e+18  ...           0.138405       0.496897
        2            1.388534e+18            1.388535e+18  ...           0.661271       0.497827
        3            1.388534e+18            1.388536e+18  ...           0.607850       0.298108
        4            1.388534e+18            1.388536e+18  ...           0.944871       0.063213
        ..                    ...                     ...  ...                ...            ...
        994          1.388794e+18            1.388872e+18  ...           0.245157       0.814368
        995          1.388794e+18            1.388873e+18  ...           0.877496       0.698151
        996          1.388794e+18            1.388873e+18  ...           0.297747       0.260010
        997          1.388794e+18            1.388873e+18  ...           0.582963       0.226867
        998          1.388794e+18            1.388874e+18  ...           0.273372       0.841499

 [999 rows x 17 columns],

    'sessions': action  ?action  action_type  ...  device_type  secs_elapsed  ?secs_elapsed
       0      0.336516        1     0.152740  ...     0.726133           319              1
       1      0.601945        1     0.500604  ...     0.766118         67753              1
       2      0.346697        1     0.162124  ...     0.785091           301              1
       3      0.573250        1     0.555400  ...     0.766361         22141              1
       4      0.363343        1     0.223726  ...     0.702190           435              1
       ...         ...      ...          ...  ...          ...           ...            ...
       69491  0.718932        1     0.280253  ...     0.696064            65              1
       69492  0.448234        1     0.729959  ...     0.719088           198              1
       69493  0.722997        1     0.902251  ...     0.651095          4626              1
       69494  0.899482        1     0.740102  ...     0.728140           151              1
       69495  0.426971        1     0.726414  ...     0.445924           396              1

 [69496 rows x 9 columns]
}
```

### 2. Reverse the transformed data

In order to reverse the transformed data call `reverse_transform` of the transformer
with the transformed data.

```python
reverse_transformed = ht.reverse_transform(tables=transformed)
```

The output will be the transformed data reversed:

```
{
    'users': date_account_created timestamp_first_active  ... first_device_type  first_browser
         0             2014-01-01         20140101010936  ...   Windows Desktop         Chrome
         1             2014-01-01         20140101011558  ...       Mac Desktop        Firefox
         2             2014-01-01         20140101011639  ...   Windows Desktop        Firefox
         3             2014-01-01         20140101012146  ...   Windows Desktop         Chrome
         4             2014-01-01         20140101012619  ...            iPhone      -unknown-
         ..                   ...                    ...  ...               ...            ...
         994           2014-01-04         20140104225038  ...       Mac Desktop         Safari
         995           2014-01-04         20140104225645  ...              iPad  Mobile Safari
         996           2014-01-04         20140104225702  ...       Mac Desktop         Chrome
         997           2014-01-04         20140104230004  ...   Windows Desktop         Chrome
         998           2014-01-04         20140104231731  ...       Mac Desktop         Safari

 [999 rows x 14 columns],

    'sessions':   action action_type            action_detail      device_type  secs_elapsed
 0                lookup   -unknown-                -unknown-  Windows Desktop           319
 1        search_results       click      view_search_results  Windows Desktop         67753
 2                lookup   -unknown-                -unknown-  Windows Desktop           301
 3        search_results       click      view_search_results  Windows Desktop         22141
 4                lookup   -unknown-                -unknown-  Windows Desktop           435
 ...                 ...         ...                      ...              ...           ...
 69491              show   -unknown-                -unknown-  Windows Desktop            65
 69492       personalize        data  wishlist_content_update  Windows Desktop           198
 69493              show        view                       p3  Windows Desktop          4626
 69494  similar_listings        data         similar_listings  Windows Desktop           151
 69495       personalize        data  wishlist_content_update      Mac Desktop           396

 [69496 rows x 5 columns]
}
```
