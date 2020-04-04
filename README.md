<img width="250" src="https://mlsploit.github.io/static/img/mlsploit-logo.png">

[![Build Status](https://travis-ci.com/mlsploit/mlsploit-py.svg?token=6hiBszjT7tvPxkvQ4Bx4&branch=master)](https://travis-ci.com/mlsploit/mlsploit-py)
[![Code Coverage](https://img.shields.io/codecov/c/gh/mlsploit/mlsploit-py)](https://codecov.io/gh/mlsploit/mlsploit-py)
[![Updates](https://pyup.io/repos/github/mlsploit/mlsploit-py/shield.svg)](https://pyup.io/repos/github/mlsploit/mlsploit-py/)
[![Python 3](https://pyup.io/repos/github/mlsploit/mlsploit-py/python-3-shield.svg)](https://pyup.io/repos/github/mlsploit/mlsploit-py/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> This package contains utilities for developing an MLsploit python module.

## Setup
To install this package on your system using [pip](https://pip.pypa.io/en/stable/), run the following command:
```bash
$ pip install mlsploit-py
```

## Usage
This package supports automatic loading of MLsploit jobs inside your module.

```python
# your_mlsploit_module.py

from mlsploit import Job

# load input files and user defined options
Job.initialize()

function_name = Job.function_name # the function name the user wants to run
input_file_items = Job.input_file_items # you can load input_file_items[i].path
options = Job.options # can access options.option1, option.option2 for function_name

# ...
# do stuff based on the information above ...
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

# you can add several output files in the above manner

# when you're done done, don't forget to commit the output!
Job.commit_output() # mlsploit-py will take care of the rest
```


You can also use this package to programmatically create your MLsploit module configuration file (`mlsploit_module.yaml`)
instead of editing it manually. This will also validate your MLsploit module for errors.
For more information on the configuration file schema,
[see here](https://github.com/mlsploit/mlsploit-py/blob/master/mlsploit/_auxiliary/mlsploit_module.schema).

```python
# make_dummy_module.py

from mlsploit import Module

module = Module.build(
    display_name='Dummy Module',
    tagline='This is a dummy module!',
    doctxt="""Long documentation for this module will go here...""",
    icon_url='https://somedomain.org/icon.jpg')

function = module.build_function(
    name='Test Function',
    doctxt="""Some long ducumentation of Test Function...""",
    creates_new_files=True,
    modifies_input_files=False,
    expected_filetype='txt',
    optional_filetypes=['rtf', 'ans'])
function.add_option(
    name='option1',
    type='str',
    doctxt="""Some long ducumentation of option1...""",
    required=True)
function.add_output_tag(name='tag1', type='str')
function.add_output_tag(name='tag2', type='int')

# you can add several functions and function options in the above manner

# once you're done, the following command will save the mlsploit_module.yaml
# in the same directory as this file
module.save()
```
