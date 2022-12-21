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
[![Slack](https://img.shields.io/badge/Community-Slack-blue?style=plastic&logo=slack)](https://bit.ly/sdv-slack-invite)

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

RDT (Reversible Data Transforms) is a Python library that transforms raw data into fully numerical
data, ready for data science. The transforms are reversible, allowing you to convert from numerical
data back into your original format.

<img align="center" src="https://github.com/sdv-dev/SDV/blob/master/docs/images/rdt_main_tranformation.png"></img>


# Install

Install **RDT** using ``pip``  or ``conda``. We recommend using a virtual environment to avoid
conflicts with other software on your device.

```bash
pip install rdt
```

```bash
conda install -c conda-forge rdt
```

For more information about using reversible data transformations, visit the [RDT Documentation](https://docs.sdv.dev/rdt).


# Quickstart

In this short series of tutorials we will guide you through a series of steps that will
help you getting started using **RDT** to transform columns, tables and datasets.

## Load the demo data

After you have installed RDT, you can get started using the demo dataset.

```python3
from rdt import get_demo

customers = get_demo()
```

This dataset contains some randomly generated values that describe the customers of an online
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

The ``HyperTransformer`` is capable of transforming multi-column datasets.

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
        "last_login": "UnixTimestampEncoder()",
        "email_optin": "BinaryEncoder()",
        "credit_card": "FrequencyEncoder()",
        "age": "FloatFormatter()",
        "dollars_spent": "FloatFormatter()"
    }
}
```

The `sdtypes` dictionary describes the semantic data types of each of your columns and the
`transformers` dictionary describes which transformer to use for each column. You can customize the
transformers and their settings. (See the [Transformers Glossary](https://docs.sdv.dev/rdt/transformers-glossary/browse-transformers) for more information).

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

The ``HyperTransformer`` applied the assigned transformer to each individual column. Each column
now contains fully numerical data that you can use for your project!

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

# What's Next?

To learn more about reversible data transformations, visit the [RDT Documentation](https://docs.sdv.dev/rdt).


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
