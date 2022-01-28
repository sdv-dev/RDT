<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/RDT.svg)](https://pypi.python.org/pypi/RDT)
[![Unit Tests](https://github.com/sdv-dev/RDT/actions/workflows/unit.yml/badge.svg)](https://github.com/sdv-dev/RDT/actions/workflows/unit.yml)
[![Downloads](https://pepy.tech/badge/rdt)](https://pepy.tech/project/rdt)
[![Coverage Status](https://codecov.io/gh/sdv-dev/RDT/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/RDT)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/RDT">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/RDT-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

**RDT** is a Python library used to transform data for data science libraries and preserve
the transformations in order to revert them as needed.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the SDV Website for more information about the project.    |
| :orange_book: **[SDV Blog]**                  | Regular publshing of useful content about Synthetic Data Generation. |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :scroll: **[License]**                        | The entire ecosystem is published under the MIT License.             |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                             |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |
| [![][MyBinder Logo] **Tutorials**][Tutorials] | Run the SDV Tutorials in a Binder environment.                       |

[Website]: https://sdv.dev
[SDV Blog]: https://sdv.dev/blog
[Documentation]: https://sdv.dev/SDV
[Repository]: https://github.com/sdv-dev/RDT
[License]: https://github.com/sdv-dev/RDT/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw
[MyBinder Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/mybinder.png
[Tutorials]: https://mybinder.org/v2/gh/sdv-dev/SDV/master?filepath=tutorials

# Install

**RDT** is part of the **SDV** project and is automatically installed alongside it. For
details about this process please visit the [SDV Installation Guide](
https://sdv.dev/SDV/getting_started/install.html)

Optionally, **RDT** can also be installed as a standalone library using the following commands:

**Using `pip`:**

```bash
pip install rdt
```

**Using `conda`:**

```bash
conda install -c conda-forge rdt
```

For more installation options please visit the [RDT installation Guide](INSTALL.md)


# Quickstart

In this short series of tutorials we will guide you through a series of steps that will
help you getting started using **RDT** to transform columns, tables and datasets.

## Transforming a column

In this first guide, you will learn how to use **RDT** in its simplest form, transforming
a single column loaded as a `pandas.DataFrame` object.

### 1. Load the demo data

You can load some demo data using the `rdt.get_demo` function, which will return some random
data for you to play with.

```python3
from rdt import get_demo

data = get_demo()
```

This will return a `pandas.DataFrame` with 10 rows and 4 columns, one of each data type supported:

```
   0_int    1_float 2_str          3_datetime
0   38.0  46.872441     b 2021-02-10 21:50:00
1   77.0  13.150228   NaN 2021-07-19 21:14:00
2   21.0        NaN     b                 NaT
3   10.0  37.128869     c 2019-10-15 21:39:00
4   91.0  41.341214     a 2020-10-31 11:57:00
5   67.0  92.237335     a                 NaT
6    NaN  51.598682   NaN 2020-04-01 01:56:00
7    NaN  42.204396     c 2020-03-12 22:12:00
8   68.0        NaN     c 2021-02-25 16:04:00
9    7.0  31.542918     a 2020-07-12 03:12:00
```

Notice how the data is random, so your output might look a bit different. Also notice how
RDT introduced some null values randomly.

### 2. Load the transformer

In this example we will use the datetime column, so let's load a `DatetimeTransformer`.

```python3
from rdt.transformers import DatetimeTransformer

transformer = DatetimeTransformer()
```

### 3. Fit the Transformer

Before being able to transform the data, we need the transformer to learn from it.

We will do this by calling its `fit` method passing the column that we want to transform.

```python3
transformer.fit(data, columns=['3_datetime'])
```

### 4. Transform the data

Once the transformer is fitted, we can pass the data again to its `transform` method in order
to get the transformed version of the data.

```python3
transformed = transformer.transform(data)
```

The output will be a `numpy.ndarray` with two columns, one with the datetimes transformed
to integer timestamps, and another one indicating with 1s which values were null in the
original data.

```
array([[1.61299380e+18, 0.00000000e+00],
       [1.62672924e+18, 0.00000000e+00],
       [1.59919923e+18, 1.00000000e+00],
       [1.57117554e+18, 0.00000000e+00],
       [1.60414542e+18, 0.00000000e+00],
       [1.59919923e+18, 1.00000000e+00],
       [1.58570616e+18, 0.00000000e+00],
       [1.58405112e+18, 0.00000000e+00],
       [1.61426904e+18, 0.00000000e+00],
       [1.59452352e+18, 0.00000000e+00]])
```

### 5. Revert the column transformation

In order to revert the previous transformation, the transformed data can be passed to
the `reverse_transform` method of the transformer:

```python3
reversed_data = transformer.reverse_transform(transformed)
```

The output will be a `pandas.Series` containing the reverted values, which should be exactly
like the original ones.

```
0   2021-02-10 21:50:00
1   2021-07-19 21:14:00
2                   NaT
3   2019-10-15 21:39:00
4   2020-10-31 11:57:00
5                   NaT
6   2020-04-01 01:56:00
7   2020-03-12 22:12:00
8   2021-02-25 16:04:00
9   2020-07-12 03:12:00
dtype: datetime64[ns]
```

## Transforming a table

Once we know how to transform a single column, we can try to go the next level and transform
a table with multiple columns.

### 1. Load the HyperTransformer

In order to manuipulate a complete table we will need to load a `rdt.HyperTransformer`.

```python3
from rdt import HyperTransformer

ht = HyperTransformer()
```

### 2. Fit the HyperTransformer

Just like the transfomer, the HyperTransformer needs to be fitted before being able to transform
data.

This is done by calling its `fit` method passing the `data` DataFrame.

```python3
ht.fit(data)
```

### 3. Transform the table data

Once the HyperTransformer is fitted, we can pass the data again to its `transform` method in order
to get the transformed version of the data.

```python3
transformed = ht.transform(data)
```

The output, will now be another `pandas.DataFrame` with the numerical representation of our
data.

```
    0_int  0_int#1    1_float  1_float#1  2_str    3_datetime  3_datetime#1
0  38.000      0.0  46.872441        0.0   0.70  1.612994e+18           0.0
1  77.000      0.0  13.150228        0.0   0.90  1.626729e+18           0.0
2  21.000      0.0  44.509511        1.0   0.70  1.599199e+18           1.0
3  10.000      0.0  37.128869        0.0   0.15  1.571176e+18           0.0
4  91.000      0.0  41.341214        0.0   0.45  1.604145e+18           0.0
5  67.000      0.0  92.237335        0.0   0.45  1.599199e+18           1.0
6  47.375      1.0  51.598682        0.0   0.90  1.585706e+18           0.0
7  47.375      1.0  42.204396        0.0   0.15  1.584051e+18           0.0
8  68.000      0.0  44.509511        1.0   0.15  1.614269e+18           0.0
9   7.000      0.0  31.542918        0.0   0.45  1.594524e+18           0.0
```

### 4. Revert the table transformation

In order to revert the transformation and recover the original data from the transformed one,
we need to call `reverse_transform` method of the `HyperTransformer` instance passing it the
transformed data.

```python3
reversed_data = ht.reverse_transform(transformed)
```

Which should output, again, a table that looks exactly like the original one.

```
   0_int    1_float 2_str          3_datetime
0   38.0  46.872441     b 2021-02-10 21:50:00
1   77.0  13.150228   NaN 2021-07-19 21:14:00
2   21.0        NaN     b                 NaT
3   10.0  37.128869     c 2019-10-15 21:39:00
4   91.0  41.341214     a 2020-10-31 11:57:00
5   67.0  92.237335     a                 NaT
6    NaN  51.598682   NaN 2020-04-01 01:56:00
7    NaN  42.204396     c 2020-03-12 22:12:00
8   68.0        NaN     c 2021-02-25 16:04:00
9    7.0  31.542918     a 2020-07-12 03:12:00
```

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://sdv.dev/SDV/getting_started/install.html) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
