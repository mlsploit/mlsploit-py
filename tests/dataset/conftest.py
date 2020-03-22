# pylint: disable=redefined-outer-name

from pathlib import Path
import random
import string

import numpy as np
from numpy import random as np_random
import pytest

from mlsploit.dataset import Dataset

from .constants import *


@pytest.fixture
def tmp_dataset_path(tmp_path) -> Path:
    dataset_dir = tmp_path/'testdata'
    dataset_dir.mkdir()
    dataset_path = dataset_dir/RECOMMENDED_FILENAME
    return dataset_path


@pytest.fixture
def make_random_valid_identifier():
    def __make_random_valid_identifier():
        valid_chars = string.ascii_letters + string.digits + '_'

        identifier = random.choice(string.ascii_lowercase)
        identifier += ''.join([random.choice(valid_chars)
                               for _ in range(random.randint(1, 5))])
        return identifier

    return __make_random_valid_identifier


@pytest.fixture
def make_random_invalid_identifier(make_random_valid_identifier):
    def __make_random_valid_identifier():
        invalid_chars = string.punctuation.replace('_', '')

        identifier = make_random_valid_identifier()
        identifier += ''.join([random.choice(invalid_chars)
                               for _ in range(random.randint(1, 5))])
        identifier = list(identifier)
        random.shuffle(identifier)
        return ''.join(identifier)

    return __make_random_valid_identifier


@pytest.fixture
def make_random_item_attr(make_random_valid_identifier):
    def __make_random_item_attr():
        name = make_random_valid_identifier()
        shape = None if random.random() < 0.2 \
            else tuple(random.randint(1, 50)
                       for _ in range(random.randint(1, 4)))
        dtype = random.choice([str, int, float, bool, np.uint8, np.float32])

        return Dataset.ItemAttr(
            name=name,
            shape=shape,
            dtype=dtype)

    return __make_random_item_attr


@pytest.fixture
def make_random_data():
    def __make_random_data(shape, dtype):
        data = np_random.random(shape)
        data = data * random.randint(1, 1000)
        data = data.astype(dtype) \
            if isinstance(data, np.ndarray) \
            else dtype(data)
        return data

    return __make_random_data


@pytest.fixture
def make_random_item_dict(make_random_data):
    def __make_random_item(item_attrs):
        item_dict = dict()
        for item_attr in item_attrs:
            attr_name = item_attr.name
            attr_shape = item_attr.shape
            attr_dtype = item_attr.dtype

            item_dict[attr_name] = \
                make_random_data(attr_shape, attr_dtype)
        return item_dict

    return __make_random_item


@pytest.fixture
def make_random_item_dicts(make_random_item_dict):
    def __make_random_item_dicts(item_attrs):
        num_items = random.randint(1, 20)
        return [make_random_item_dict(item_attrs)
                for _ in range(num_items)]

    return __make_random_item_dicts


@pytest.fixture
def random_item_attrs(make_random_item_attr):
    num_item_attrs = random.randint(1, 5)
    item_attrs = [make_random_item_attr() for _ in range(num_item_attrs)]
    item_attrs = {it.name: it for it in item_attrs} # drop duplicates
    return list(item_attrs.values())


@pytest.fixture
def random_metadata_dict(make_random_valid_identifier):
    keys = [make_random_valid_identifier()
            for _ in range(random.randint(1, 10))]

    metadata = dict()
    for k in keys:
        type_ = random.choice([str, int, float, bool])

        if type_ is str:
            metadata[k] = ''.join([random.choice(string.printable)
                                   for _ in range(random.randint(1, 100))])

        elif type_ is int:
            metadata[k] = random.randint(1, 1000)

        elif type_ is float:
            metadata[k] = random.random() * random.randint(1, 1000)

        elif type_ is bool:
            metadata[k] = random.choice([True, False])

    return metadata


@pytest.fixture
def random_empty_dataset(tmp_dataset_path, random_item_attrs, random_metadata_dict):
    return Dataset(tmp_dataset_path, random_item_attrs,
                   metadata=random_metadata_dict)


@pytest.fixture
def random_dataset_with_random_data(make_random_item_dict, random_empty_dataset):
    dataset = random_empty_dataset
    num_items = random.randint(1, 20)
    for _ in range(num_items):
        dataset.add_item(**make_random_item_dict(dataset.item_attrs))

    return dataset
