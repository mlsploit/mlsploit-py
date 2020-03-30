# pylint: disable=redefined-outer-name

import keyword
from pathlib import Path
import random
import string

import numpy as np
from numpy import random as np_random
import pytest

from mlsploit.dataset import Dataset
from mlsploit.dataset.base import Feature

from .constants import *


@pytest.fixture
def tmp_dataset_path(tmp_path) -> Path:
    dataset_dir = tmp_path / "testdata"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / RECOMMENDED_FILENAME
    return dataset_path


@pytest.fixture
def make_random_valid_identifier():
    def __make_random_valid_identifier():
        valid_chars = string.ascii_letters + string.digits + "_"

        identifier = random.choice(string.ascii_lowercase)
        identifier += "".join(
            [random.choice(valid_chars) for _ in range(random.randint(1, 5))]
        )
        return (
            identifier
            if not keyword.iskeyword(identifier)
            else __make_random_valid_identifier()
        )

    return __make_random_valid_identifier


@pytest.fixture
def make_random_invalid_identifier(make_random_valid_identifier):
    def __make_random_valid_identifier():
        invalid_chars = string.punctuation.replace("_", "")

        identifier = make_random_valid_identifier()
        identifier += "".join(
            [random.choice(invalid_chars) for _ in range(random.randint(1, 5))]
        )
        identifier = list(identifier)
        random.shuffle(identifier)
        return "".join(identifier)

    return __make_random_valid_identifier


@pytest.fixture
def make_random_feature(make_random_valid_identifier):
    def __make_random_feature():
        name = make_random_valid_identifier()
        shape = (
            None
            if random.random() < 0.2
            else tuple(random.randint(1, 50) for _ in range(random.randint(1, 4)))
        )
        dtype = random.choice([str, int, float, bool, np.uint8, np.float32])

        return Feature(name=name, shape=shape, dtype=dtype)

    return __make_random_feature


@pytest.fixture
def make_random_data():
    def __make_random_data(shape, dtype):
        shape = (1,) + shape if isinstance(shape, tuple) else (1,)

        data = np_random.random(shape)
        data = data * random.randint(1, 1000)
        data = data.astype(dtype)
        return data[0]

    return __make_random_data


@pytest.fixture
def make_random_item_dict(make_random_data):
    def __make_random_item(features):
        item_dict = dict()
        for feature in features:
            feat_name = feature.name
            feat_shape = feature.shape
            feat_dtype = feature.dtype

            item_dict[feat_name] = make_random_data(feat_shape, feat_dtype)
        return item_dict

    return __make_random_item


@pytest.fixture
def make_random_item_dicts(make_random_item_dict):
    def __make_random_item_dicts(features):
        num_items = random.randint(1, 20)
        return [make_random_item_dict(features) for _ in range(num_items)]

    return __make_random_item_dicts


@pytest.fixture
def random_features(make_random_feature):
    num_features = random.randint(1, 5)
    features = [make_random_feature() for _ in range(num_features)]
    features = {f.name: f for f in features}  # drop duplicates
    return list(features.values())


@pytest.fixture
def random_metadata_dict(make_random_valid_identifier):
    keys = [make_random_valid_identifier() for _ in range(random.randint(1, 10))]

    metadata = dict()
    for k in keys:
        type_ = random.choice([str, int, float, bool])

        if type_ is str:
            metadata[k] = "".join(
                [random.choice(string.printable) for _ in range(random.randint(1, 100))]
            )

        elif type_ is int:
            metadata[k] = random.randint(1, 1000)

        elif type_ is float:
            metadata[k] = random.random() * random.randint(1, 1000)

        elif type_ is bool:
            metadata[k] = random.choice([True, False])

    return metadata


@pytest.fixture
def random_empty_dataset(tmp_dataset_path, random_features, random_metadata_dict):

    dataset = Dataset.build(tmp_dataset_path).with_metadata(**random_metadata_dict)
    for feature in random_features:
        dataset.add_feature(**feature.dict())

    return dataset.conclude_build()


@pytest.fixture
def random_dataset_with_random_data(make_random_item_dict, random_empty_dataset):

    dataset = random_empty_dataset
    num_items = random.randint(1, 20)
    for _ in range(num_items):
        dataset.add_item(**make_random_item_dict(dataset.features))

    return dataset
