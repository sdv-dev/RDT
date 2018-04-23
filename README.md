# data-prep
This a a python library used to clean up and prepare data for use with other data science libraries.
## Installation
You can create a virtual environment and install the dependencies using the following commands.
```bash
$ virtualenv venv --no-site-packages
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Usage
This library is used to apply desired transformations to individual tables or entire datasets all at once, with the goal of getting completely numeric tables as the output. The desired transformations can be specified at the column level, or dataset level. For example, you can apply a datetime transformation to only select columns, or you can specify that you want every datetime column in the dataset to go through that transformation.
### Downloading demo data
If you don't have data to work with right away, you can download our demo data by running the following command from the root directory of this project.
```bash
$ python demo_downloader.py
```
### Transforming a column
The base class of this library is the BaseTransformer class. This class provides method to fit a transformer to your data and transform it, a method to transform new data with an already fitted transformer and a method to reverse a transform and get data that looks like the original input. Each transformer class inherits from the BaseTransformer class, and thus has all these methods. 

Transformers take in a column and the meta data for that column as an input. Below we will demonstrate how to use a datetime transformer to transform and reverse transform a column.

First load the data. 
```bash
>>> from dataprep.transformers.DTTransformer import *
>>> from dataprep.utils import *
>>> col, col_meta = get_col_info('users', 'date_account_created', 'demo/Airbnb_demo_meta.json')
>>> print(col)
0      2014-01-01
1      2014-01-01
2      2014-01-01
3      2014-01-01
4      2014-01-01
5      2014-01-01
6      2014-01-01
...
>>> print(col_meta)
{'type': 'datetime', 'name': 'date_account_created', 'uniques': 1634, 'format': '%Y-%m-%d'}
```
Now we can transform the column.
```bash
>>> transformer = DTTransformer()
>>> transformed_data = transformer.fit_transform(col, col_meta)
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
If you want to reverse the transformation and get the original data back, you can run the following command.
```bash
>>> reversed = transformer.reverse_transform(transformed_data['date_account_created'], col_meta)
>>> print(reversed)
    date_account_created
0             2014-01-01
1             2014-01-01
2             2014-01-01
3             2014-01-01
4             2014-01-01
```
### Transforming a table
You can also transform an entire table using the HyperTransformer class. Again, we can start by loading the data.
```bash
>>> from dataprep.hyper_transformer import *
>>> from dataprep.utils import *
>>> meta_file = 'data/Airbnb_demo_meta.json'
>>> table_dict = get_table_dict(meta_file)
>>> table, table_meta = table_dict['users']
```
Now you can pass a list of the desired transformers into the fit_transform_table function to transform the whole table.
```bash
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
You can then reverse transform the output to get a table in the original format, but it will only contain the columns corresponding to those that were transformed (ie. numeric columns).
```bash
>>> reversed = ht.reverse_transform_table(transformed, table_meta)
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
The hyper transformer is also capable of transforming all of the tables specified in your meta.json at once.
```bash
>>> from dataprep.hyper_transformer import *
>>> meta_file = 'data/Airbnb_demo_meta.json'
>>> ht = HyperTransformer(meta_file)
>>> tl = ['DTTransformer', 'NumberTransformer']
>>> transformed = ht.hyper_fit_transform(transformer_list=tl)
>>> 
{'sessions':        secs_elapsed  ?secs_elapsed
0             319.0              1
1           67753.0              1
2             301.0              1
3           22141.0              1
4             435.0              1
5            7703.0              1
6             115.0              1
7             831.0              1
...
>>> reversed = ht.hyper_reverse_transform(tables=transformed)
>>> print(reversed)
{'sessions':        secs_elapsed
0             319.0
1           67753.0
2             301.0
3           22141.0
4             435.0
5            7703.0
6             115.0
7             831.0
8           20842.0
...
```