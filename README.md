# MLsploit Module Utilities

[![Build Status](https://travis-ci.com/mlsploit/mlsploit-py.svg?token=6hiBszjT7tvPxkvQ4Bx4&branch=master)](https://travis-ci.com/mlsploit/mlsploit-py)
[![Code Coverage](https://img.shields.io/codecov/c/gh/mlsploit/mlsploit-py)](https://codecov.io/gh/mlsploit/mlsploit-py)
[![Updates](https://pyup.io/repos/github/mlsploit/mlsploit-py/shield.svg)](https://pyup.io/repos/github/mlsploit/mlsploit-py/)
[![Python 3](https://pyup.io/repos/github/mlsploit/mlsploit-py/python-3-shield.svg)](https://pyup.io/repos/github/mlsploit/mlsploit-py/)

This package contains utilities for developing an MLsploit python module.

## Setup
To install this package on your system using [pip](https://pip.pypa.io/en/stable/), run the following command:
```bash
$ pip install git+https://github.com/mlsploit/mlsploit-py
```
This will fetch the package from this repository and install it locally. We're also working on releasing a [PyPI](https://pypi.org/) distribution soon.

## Usage
This package support automatic loading of MLsploit jobs inside your module.
```python
# your_mlsploit_module.py

from mlsploit import Job

# load input files and user defined options
Job.initialize()

function_name = Job.function_name # the function name the user wants to run
input_file_items = Job.input_file_items # you can load input_file_items[i].path
options = Job.options # can access options.option1, option.option2 for function_name

# ...
# do stuff based on the information above
# ...

# when you're done, you can reserve output files
# that will be uploaded to MLsploit
output_file_item = Job.reserve_output_file_item(
    output_file_name='output_file.txt', is_new_file=True)

with open(output_file_item.path, 'w') as f:
    pass # ... write data to your output file here

# you can also add tags to your output file
# as defined in your module schema
output_file_item.add_tag(name='tagname', value='tagvalue')

# you can add as many output files in the above manner as you wish
# when you're done done, don't forget to commit the output!
Job.commit_output() # this package will take care of the rest
