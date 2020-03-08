import json
from pathlib import Path

import pytest

from mlsploit import Job, Module
from mlsploit.paths import JobPaths, ModulePaths

from .constants import *


@pytest.fixture
def tmp_module_dir(tmp_path) -> Path:
    module_dir = tmp_path/'testmodule'
    module_dir.mkdir()
    ModulePaths.set_module_dir(module_dir)
    yield module_dir
    ModulePaths.reset_module_dir()


@pytest.fixture
def tmp_job_dir(tmp_path) -> Path:
    job_dir = tmp_path/'testjob'
    job_dir.mkdir()
    JobPaths.set_job_dir(job_dir)
    JobPaths().input_dir.mkdir()
    JobPaths().output_dir.mkdir()
    yield job_dir
    JobPaths.reset_job_dir()


@pytest.fixture
def degenerate_module() -> Module:
    return Module.build(
        display_name=MODULE_DISPLAY_NAME,
        tagline=MODULE_TAGLINE, doctxt=MODULE_DOCTXT,
        icon_url=MODULE_ICON_URL.format(extension='jpg'))


@pytest.fixture
def dummy_module(degenerate_module) -> Module:
    m = degenerate_module
    f = m.add_function(
        name=FUNCTION_NAME,
        doctxt=FUNCTION_DOCTXT,
        creates_new_files=True,
        modifies_input_files=False,
        expected_filetype=FUNCTION_EXPECTED_FILETYPE,
        optional_filetypes=FUNCTION_OPTIONAL_FILETYPES)
    f.add_option(name=OPTION_NAME,
                 type=OPTION_TYPE,
                 doctxt=OPTION_DOCTXT,
                 required=True)
    f.add_output_tag(name=TAG_NAME, type=TAG_TYPE)
    return m


@pytest.fixture
def dummy_input_document_dict(tmp_job_dir) -> dict:
    input_file_name = 'test.txt'
    with open(JobPaths().input_dir/input_file_name, 'w'):
        pass
    return {'name': FUNCTION_NAME,
            'num_files': 1,
            'files': [input_file_name],
            'options': {OPTION_NAME: 'dummy_option_value'},
            'tags': [{TAG_NAME: 123}]}


@pytest.fixture
def initialized_job(dummy_module, dummy_input_document_dict,
                    tmp_module_dir, tmp_job_dir):

    dummy_module.save()
    with open(JobPaths().input_data_file, 'w') as f:
        json.dump(dummy_input_document_dict, f)
    Job.initialize()
    yield Job
    Job.reset()
