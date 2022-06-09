<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-3%20--%20Alpha-yellow)](https://pypi.org/search/?q=&o=&c=Development+Status+%3A%3A+3+-+Alpha)
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
| :keyboard: **[Development Status]**           | This software is in its Alpha stage.                                 |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |
| [![][Google Colab Logo] **Tutorials**][Tutorials] | Run the RDT Tutorials in a notebook.                             |

[Website]: https://sdv.dev
[SDV Blog]: https://sdv.dev/blog
[Documentation]: https://docs.sdv.dev/rdt
[Repository]: https://github.com/sdv-dev/RDT
[License]: https://github.com/sdv-dev/RDT/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?q=&o=&c=Development+Status+%3A%3A+3+-+Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://bit.ly/sdv-slack-invite
[Google Colab Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/google_colab.png
[Tutorials]: https://colab.research.google.com/drive/1T_3XSPPOVILATsyRV9xjQPa0hvM1vnM-?usp=sharing

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

## Load the demo data

After you have installed RDT, you can get started using the demo dataset.

```python3
from rdt import get_demo

customers = get_demo()
```

This dataset contains some randomly generated values that describes the customers of an online
marketplace. 

```
  last_login email_optin credit_card  age  dollars_spent
0 2021-06-26       False        VISA   29          99.99
1 2021-02-10       False        VISA   18            NaN
2        NaT       False        AMEX   21           2.50
3 2020-09-26        True         NaN   45          25.00
4 2020-12-22         NaN    DISCOVER   32          19.99
```

Let's transform this data so that each column is converted to full, numerical data ready for data
science.

## Creating the HyperTransformer & config

The `HyperTransformer` is capable of transforming multi-column datasets.

```python3
from rdt import HyperTransformer

ht = HyperTransformer()
```

The `HyperTransformer` needs to know about the columns in your dataset and which transformers to
apply to each. These are described by a config. We can ask the `HyperTransformer` to automatically
detect it based on the data we plan to use.

```python3
ht.detect_initial_config(data=customers)
```

This will create and set the config.

```
Config:
{
    "sdtypes": {
        "last_login": "datetime",
        "email_optin": "boolean",
        "credit_card": "categorical",
        "age": "numerical",
        "dollars_spent": "numerical"
    },
    "transformers": {
        "last_login": "UnixTimestampEncoder(missing_value_replacement='mean')",
        "email_optin": "BinaryEncoder(missing_value_replacement='mode')",
        "credit_card": "FrequencyEncoder()",
        "age": "FloatFormatter(missing_value_replacement='mean')",
        "dollars_spent": "FloatFormatter(missing_value_replacement='mean')"
    }
}
```

The `sdtypes` dictionary describes the semantic data types of each of your columns and the
`transformers` dictionary describes which transformer to use for each column.

## Fitting & using the HyperTransformer 

The `HyperTransformer` references the config while learning the data during the `fit` stage.

```python3
ht.fit(customers)
```

Once the transformer is fit, it's ready to use. Use the transform method to transform all columns
of your dataset at once.

```python3
transformed_data = ht.transform(customers)
```

```
   last_login.value  email_optin.value  credit_card.value  age.value  dollars_spent.value
0      1.624666e+18                0.0                0.2         29                99.99
1      1.612915e+18                0.0                0.2         18                36.87
2      1.611814e+18                0.0                0.5         21                 2.50
3      1.601078e+18                1.0                0.7         45                25.00
4      1.608595e+18                0.0                0.9         32                19.99
```

The `HyperTransformer` applied the assigned transformer to each individual column. Each column now
contains fully numerical data that you can use for your project!

When you're done with your project, you can also transform the data back to the original format
using the `reverse_transform` method.

```python3
original_format_data = ht.reverse_transform(transformed_data)
```

```
  last_login email_optin credit_card  age  dollars_spent
0        NaT       False        VISA   29          99.99
1 2021-02-10       False        VISA   18            NaN
2        NaT       False        AMEX   21            NaN
3 2020-09-26        True         NaN   45          25.00
4 2020-12-22       False    DISCOVER   32          19.99
```

## Transforming a single column

It is also possible to transform a single column of a `pandas.DataFrame`. To do this,
follow the following steps.

### Load the transformer

In this example we will use the datetime column, so let's load a `UnixTimestampEncoder`.

```python3
from rdt.transformers import UnixTimestampEncoder

transformer = UnixTimestampEncoder()
```

### Fit the Transformer

Before being able to transform the data, we need the transformer to learn from it.

We will do this by calling its `fit` method passing the column that we want to transform.

```python3
transformer.fit(customers, column='last_login')
```

### Transform the data

Once the transformer is fitted, we can pass the data again to its `transform` method in order
to get the transformed version of the data.

```python3
transformed = transformer.transform(customers)
```

The output will be a `pandas.DataFrame` similar to the input data, except with the original
datetime column replaced with `last_login.value`.

```
  email_optin credit_card  age  dollars_spent  last_login.value
0       False        VISA   29          99.99      1.624666e+18
1       False        VISA   18            NaN      1.612915e+18
2       False        AMEX   21           2.50               NaN
3        True         NaN   45          25.00      1.601078e+18
4         NaN    DISCOVER   32          19.99      1.608595e+18
```

### Revert the column transformation

In order to revert the previous transformation, the transformed data can be passed to
the `reverse_transform` method of the transformer:

```python3
reversed_data = transformer.reverse_transform(transformed)
```

The output will be a `pandas.DataFrame` containing the reverted values, which should be exactly
like the original ones, except for the order of the columns.

```
  email_optin credit_card  age  dollars_spent last_login
0       False        VISA   29          99.99 2021-06-26
1       False        VISA   18            NaN 2021-02-10
2       False        AMEX   21           2.50        NaT
3        True         NaN   45          25.00 2020-09-26
4         NaN    DISCOVER   32          19.99 2020-12-22
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
