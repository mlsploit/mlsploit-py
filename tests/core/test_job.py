import json
from pathlib import Path

from devtools import debug
import pytest

from mlsploit import Job
from mlsploit.paths import JobPaths

from .constants import *


def _save_input_document_dict(input_document_dict: dict) -> Path:
    path = JobPaths().input_data_file
    with open(path, 'w') as f:
        json.dump(input_document_dict, f)
    return path


# TODO: test validation logic

def test_job_constructor():
    with pytest.raises(NotImplementedError) as excinfo:
        Job()
    assert 'use Job.initialize instead' in str(excinfo)


def test_job_initialize(dummy_module, dummy_input_document_dict,
                        tmp_module_dir, tmp_job_dir):

    debug(tmp_module_dir)
    debug(tmp_job_dir)

    dummy_module.save()

    debug(
        _save_input_document_dict(
            dummy_input_document_dict))

    Job.initialize()

    assert Job.module is not dummy_module
    assert Job.module == dummy_module

    assert Job.function_name == dummy_input_document_dict['name']

    for option, value in dummy_input_document_dict['options'].items():
        assert hasattr(Job.options, option)
        assert getattr(Job.options, option) == value

    assert len(Job.input_file_items) == len(dummy_input_document_dict['files'])
    for i, input_file_item in enumerate(Job.input_file_items):
        input_file_name = dummy_input_document_dict['files'][i]
        input_file_tags = dummy_input_document_dict['tags'][i]

        assert input_file_item.name == input_file_name
        assert input_file_item.tags == input_file_tags


def test_job_reserve_output_file_item(initialized_job):
    output_file_name = 'output_file.txt'
    output_tag_value = 456

    output_file_item = initialized_job.reserve_output_file_item(
        output_file_name=output_file_name,
        is_new_file=True, is_modified_file=False,
        tags={TAG_NAME: output_tag_value})

    assert output_file_item.name == output_file_name
    assert output_file_item.tags[TAG_NAME] == output_tag_value

    new_tag_value = 789
    output_file_item.add_tag(TAG_NAME, new_tag_value)
    assert output_file_item.tags[TAG_NAME] == new_tag_value


def test_job_commit_output_simple(initialized_job):
    assert not JobPaths().output_data_file.exists()
    initialized_job.commit_output()
    assert JobPaths().output_data_file.exists()


def test_job_commit_output_with_output_file(initialized_job):
    output_file_name = 'output_file.txt'
    output_tag_value = 456

    output_file_item = initialized_job.reserve_output_file_item(
        output_file_name=output_file_name,
        is_new_file=True, is_modified_file=False,
        tags={TAG_NAME: output_tag_value})
    assert not output_file_item.path.exists()

    with pytest.raises(RuntimeError) as excinfo:
        initialized_job.commit_output()
    assert 'cannot find output file item on disk' in str(excinfo)

    with open(output_file_item.path, 'w'):
        assert output_file_item.path.exists()

    assert not JobPaths().output_data_file.exists()
    initialized_job.commit_output()
    assert JobPaths().output_data_file.exists()
