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
This library is used to apply desired transformations to individual tables or entire datasets all at once, with the goal of getting completely numeric tables as the output. The desired transformations can be specified at the column level, or dataset level. For example, you can apply a datetime transformation to only select columns, r you can specify that you want every datetime column in the dataset to go through that transformation.
### Transforming a column

### Transforming a table

### Transforming a dataset