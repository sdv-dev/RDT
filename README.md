<p align="left"> 
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“Copulas” />
  <i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![][pypi-img]][pypi-url] [![][travis-img]][travis-url]

# Reversible Data Transforms

This a python library used to transform data for data science libraries and preserve the transformations in order to reverse them as needed.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/RDT

[travis-img]: https://travis-ci.org/HDI-Project/RDT.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/RDT
[pypi-img]: https://img.shields.io/pypi/v/RDT.svg
[pypi-url]: https://pypi.python.org/pypi/RDT



## Installation

### Install with pip

The simplest and recommended way to install RDT is using `pip`:

```
pip install rdt
```

### Install from sources

You can also clone the repository and install it from sources

```
git clone git@github.com:HDI-Project/RDT.git
cd RDT
pip install -e .
```

## Usage

This library is used to apply desired transformations to individual tables or entire datasets
all at once, with the goal of getting completely numeric tables as the output. The desired
transformations can be specified at the column level, or dataset level. For example, you can
apply a datetime transformation to only select columns, or you can specify that you want every
datetime column in the dataset to go through that transformation.

### Transforming a column

The base class of this library is the BaseTransformer class. This class provides method to fit
a transformer to your data and transform it, a method to transform new data with an already
fitted transformer and a method to reverse a transform and get data that looks like the original
input. Each transformer class inherits from the BaseTransformer class, and thus has all
these methods.

Transformers take in a column and the meta data for that column as an input. Below we will
demonstrate how to use a datetime transformer to transform and reverse transform a column.

First we need to decompress the demo data included in the repository by running this
command on a shell:

```
tar -xvzf examples/data/airbnb.tar.gz -C examples/data/
```

Afterwards, we can proceed to open a python interpreter and load the data

```python
>>> from rdt.transfomers import get_col_info
>>> demo_data = 'examples/data/airbnb/Airbnb_demo_meta.json'
>>> column, column_metadata = get_col_info('users', 'date_account_created', demo_data)
>>> column.head(5)
0    2014-01-01
1    2014-01-01
2    2014-01-01
3    2014-01-01
4    2014-01-01
Name: date_account_created, dtype: object

>>> column_metadata
{'name': 'date_account_created',
 'type': 'datetime',
 'format': '%Y-%m-%d',
 'uniques': 1634}

```

Now we can transform the column.

```python
>>> from rdt.transformers.DTTransformer import DTTransformer
>>> transformer = DTTransformer()
>>> transformed_data = transformer.fit_transform(column, column_metadata)
>>> transformed_data.head(5)
0                      1          1.388531e+18
1                      1          1.388531e+18
2                      1          1.388531e+18
3                      1          1.388531e+18
4                      1          1.388531e+18

```

If you want to reverse the transformation and get the original data back, you can run the
following command.

```python
>>> reverse_transformed = transformer.reverse_transform(transformed_data, column_metadata)
>>> reverse_transformed.head(5)
  date_account_created
  date_account_created
0           2014-01-01
1           2014-01-01
2           2014-01-01
3           2014-01-01
4           2014-01-01
```

### Transforming a table

You can also transform an entire table using the HyperTransformer class. Again, we can start by
loading the data.

```python
>>> from rdt.utils import get_table_dict
>>> meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
>>> table_dict = get_table_dict(meta_file)
>>> table, table_meta = table_dict['users']
```

Now you can pass a list of the desired transformers into the `fit_transform_table` function to
transform the whole table.

```python
>>> from rdt.hyper_transformer import HyperTransformer
>>> ht = HyperTransformer(meta_file)
>>> tl = ['DTTransformer', 'NumberTransformer', 'CatTransformer']
>>> transformed = ht.fit_transform_table(table, table_meta, transformer_list=tl)
>>> transformed.head(3).T
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
>>> reverse_transformed = ht.reverse_transform_table(transformed, table_meta)
>>> reverse_transformed.head(3).T
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

### Transforming a dataset

The hyper transformer is also capable of transforming all of the tables specified in your
meta.json at once.

```python
>>> from rdt.hyper_transformer import HyperTransformer
>>> meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
>>> ht = HyperTransformer(meta_file)
>>> tl = ['DTTransformer', 'NumberTransformer', 'CatTransformer']
>>> transformed = ht.fit_transform(transformer_list=tl)
>>> transformed['users'].head(3).T
                                     0             1             2
?date_account_created     1.000000e+00  1.000000e+00  1.000000e+00
date_account_created      1.388531e+18  1.388531e+18  1.388531e+18
?timestamp_first_active   1.000000e+00  1.000000e+00  1.000000e+00
timestamp_first_active    1.654000e+13  1.654000e+13  1.654000e+13
?date_first_booking       1.000000e+00  0.000000e+00  0.000000e+00
date_first_booking        1.388790e+18  0.000000e+00  0.000000e+00
?gender                   1.000000e+00  1.000000e+00  1.000000e+00
gender                    9.061832e-01  1.729590e-01  4.287514e-02
?age                      1.000000e+00  0.000000e+00  0.000000e+00
age                       6.200000e+01  3.700000e+01  3.700000e+01
?signup_method            1.000000e+00  1.000000e+00  1.000000e+00
signup_method             5.306912e-01  4.082081e-01  3.028973e-01
?signup_flow              1.000000e+00  1.000000e+00  1.000000e+00
signup_flow               4.597129e-01  4.751324e-01  5.495054e-01
?language                 1.000000e+00  1.000000e+00  1.000000e+00
language                  2.947847e-01  4.170684e-01  5.057820e-01
?affiliate_channel        1.000000e+00  1.000000e+00  1.000000e+00
affiliate_channel         9.213130e-01  4.712533e-01  8.231925e-01
?affiliate_provider       1.000000e+00  1.000000e+00  1.000000e+00
affiliate_provider        7.649791e-01  2.028804e-01  7.174262e-01
?first_affiliate_tracked  1.000000e+00  1.000000e+00  1.000000e+00
first_affiliate_tracked   3.716114e-01  6.723371e-01  3.710109e-01
?signup_app               1.000000e+00  1.000000e+00  1.000000e+00
signup_app                3.583918e-01  2.627690e-01  4.544640e-01
?first_device_type        1.000000e+00  1.000000e+00  1.000000e+00
first_device_type         6.621950e-01  3.078130e-01  7.152115e-01
?first_browser            1.000000e+00  1.000000e+00  1.000000e+00
first_browser             2.410379e-01  4.766930e-01  4.865389e-01

>>> transformed['sessions'].head(3).T
                         0             1           2
?action           1.000000      1.000000    1.000000
action            0.361382      0.597891    0.353806
?action_type      1.000000      1.000000    1.000000
action_type       0.089913      0.560351    0.046400
?action_detail    1.000000      1.000000    1.000000
action_detail     0.070212      0.852246    0.107477
?device_type      1.000000      1.000000    1.000000
device_type       0.726447      0.711231    0.710298
?secs_elapsed     1.000000      1.000000    1.000000
secs_elapsed    319.000000  67753.000000  301.000000

>>> reverse_transformed = ht.reverse_transform(tables=transformed)
>>> reverse_transformed['users'].head(3).T
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

>>> reverse_transformed['sessions'].head(3).T
                             0                    1                2
action                  lookup       search_results           lookup
action_type               None                click             None
action_detail             None  view_search_results             None
device_type    Windows Desktop      Windows Desktop  Windows Desktop
secs_elapsed               319                67753              301

```
