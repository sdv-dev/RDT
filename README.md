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

```
>>> from rdt.utils import get_col_info
>>> demo_data = 'examples/data/airbnb/Airbnb_demo_meta.json'
>>> column, column_metadata = get_col_info('users', 'date_account_created', demo_data)
>>> print(column)
0      2014-01-01
1      2014-01-01
2      2014-01-01
3      2014-01-01
4      2014-01-01
5      2014-01-01
6      2014-01-01
...
>>> print(column_metadata)
{'type': 'datetime', 'name': 'date_account_created', 'uniques': 1634, 'format': '%Y-%m-%d'}

```

Now we can transform the column.

```
>>> from rdt.transformers.DTTransformer import DTTransformer
>>> transformer = DTTransformer()
>>> transformed_data = transformer.fit_transform(column, column_metadata)
>>> print(transformed_data)
     date_account_created  ?date_account_created
0            1.388552e+18                      1
1            1.388552e+18                      1
2            1.388552e+18                      1
3            1.388552e+18                      1
4            1.388552e+18                      1
5            1.388552e+18                      1
6            1.388552e+18                      1

```

If you want to reverse the transformation and get the original data back, you can run the
following command.

```
>>> transformed_column = transformed_data['date_account_created']
>>> reversed = transformer.reverse_transform(transformed_column, column_metadata)
>>> print(reversed)
    date_account_created
0             2014-01-01
1             2014-01-01
2             2014-01-01
3             2014-01-01
4             2014-01-01
```

### Transforming a table

You can also transform an entire table using the HyperTransformer class. Again, we can start by
loading the data.

```
>>> from rdt.utils import get_table_dict
>>> meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
>>> table_dict = get_table_dict(meta_file)
>>> table, table_meta = table_dict['users']
```

Now you can pass a list of the desired transformers into the `fit_transform_table` function to
transform the whole table.

```
>>> from rdt.hyper_transformer import HyperTransformer
>>> ht = HyperTransformer(meta_file)
>>> tl = ['DTTransformer', 'NumberTransformer']
>>> transformed = ht.fit_transform_table(table, table_meta, transformer_list = tl)
>>> print(transformed)
     date_account_created  ?date_account_created  timestamp_first_active  \
0            1.388552e+18                      1            1.388553e+18
1            1.388552e+18                      1            1.388553e+18
2            1.388552e+18                      1            1.388553e+18
3            1.388552e+18                      1            1.388554e+18
4            1.388552e+18                      1            1.388554e+18
5            1.388552e+18                      1            1.388554e+18
6            1.388552e+18                      1            1.388554e+18
```

You can then reverse transform the output to get a table in the original format, but it will
only contain the columns corresponding to those that were transformed (ie. numeric columns).

```
>>> ht.reverse_transform_table(transformed, table_meta)
    date_account_created timestamp_first_active date_first_booking   age
0             2014-01-01         20140101050936         2014-01-04  62.0
1             2014-01-01         20140101051558                NaN   NaN
2             2014-01-01         20140101051639                NaN   NaN
3             2014-01-01         20140101052146                NaN   NaN
4             2014-01-01         20140101052619         2014-01-02   NaN
5             2014-01-01         20140101052626                NaN   NaN
6             2014-01-01         20140101052742         2014-01-07  32.0
```

### Transforming a dataset

The hyper transformer is also capable of transforming all of the tables specified in your
meta.json at once.

```
>>> from rdt.hyper_transformer import HyperTransformer
>>> meta_file = 'examples/data/airbnb/Airbnb_demo_meta.json'
>>> ht = HyperTransformer(meta_file)
>>> tl = ['DTTransformer', 'NumberTransformer']
>>> transformed = ht.fit_transform(transformer_list=tl)
>>> transformed['users'].head()
   ?date_account_created  date_account_created  ?timestamp_first_active  \
0                      1          1.388531e+18                        1
1                      1          1.388531e+18                        1
2                      1          1.388531e+18                        1
3                      1          1.388531e+18                        1
4                      1          1.388531e+18                        1

   timestamp_first_active  ?date_first_booking  date_first_booking ?age  age
0            1.654000e+13                    1        1.388790e+18    1   62
1            1.654000e+13                    0        0.000000e+00    0   37
2            1.654000e+13                    0        0.000000e+00    0   37
3            1.654000e+13                    0        0.000000e+00    0   37
4            1.654000e+13                    1        1.388617e+18    0   37
>>> transformed['sessions'].head()
  ?secs_elapsed  secs_elapsed
0             1           319
1             1         67753
2             1           301
3             1         22141
4             1           435
>>> reversed = ht.reverse_transform(tables=transformed)
>>> reversed['users'].head()
  date_account_created timestamp_first_active date_first_booking   age
0           2014-01-01         19700101053540         2014-01-04  62.0
1           2014-01-01         19700101053540                NaN   NaN
2           2014-01-01         19700101053540                NaN   NaN
3           2014-01-01         19700101053540                NaN   NaN
4           2014-01-01         19700101053540         2014-01-02   NaN
>>> reversed['sessions'].head()
   secs_elapsed
0         319.0
1       67753.0
2         301.0
3       22141.0
4         435.0
```
