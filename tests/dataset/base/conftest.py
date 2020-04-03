# pylint: disable=redefined-outer-name

from collections import OrderedDict
import keyword
from pathlib import Path
import random
import string

import numpy as np
from numpy import random as np_random
import pytest

from mlsploit.dataset import Dataset, Feature, Metadata
from mlsploit.dataset.base.metadata import ALLOWED_PRIMITIVES

from .constants import RECOMMENDED_FILENAME


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
    def __make_random_invalid_identifier():
        invalid_chars = string.punctuation.replace("_", "")

        identifier = make_random_valid_identifier()
        identifier += "".join(
            [random.choice(invalid_chars) for _ in range(random.randint(1, 5))]
        )
        identifier = list(identifier)
        random.shuffle(identifier)
        return "".join(identifier)

    return __make_random_invalid_identifier


@pytest.fixture
def make_random_metadata_dict(make_random_valid_identifier):
    def __make_random_metadata_dict():
        keys = [make_random_valid_identifier() for _ in range(random.randint(1, 10))]

        metadata = dict()
        for k in keys:
            type_ = random.choice(ALLOWED_PRIMITIVES)

            if type_ is str:
                metadata[k] = "".join(
                    [
                        random.choice(string.printable)
                        for _ in range(random.randint(1, 100))
                    ]
                )

            elif type_ is int:
                metadata[k] = random.randint(1, 1000)

            elif type_ is float:
                metadata[k] = random.random() * random.randint(1, 1000)

            elif type_ is bool:
                metadata[k] = random.choice([True, False])

        return metadata

    return __make_random_metadata_dict


@pytest.fixture
def make_random_metadata_object(make_random_metadata_dict):
    def __make_random_metadata_object():
        random_metadata_dict = make_random_metadata_dict()
        return Metadata(**random_metadata_dict)

    return __make_random_metadata_object


@pytest.fixture
def random_default_metadata(make_random_metadata_dict):
    random_metadata_dict = make_random_metadata_dict()
    return type("DefaultMetadata", tuple([]), random_metadata_dict)


@pytest.fixture
def make_random_feature(make_random_metadata_object):
    def __make_random_feature():
        shape = (
            None
            if random.random() < 0.2
            else tuple(random.randint(1, 50) for _ in range(random.randint(1, 4)))
        )
        #  dtype = random.choice([str, int, float, bool, np.uint8, np.float32])
        dtype = random.choice([str])
        metadata = make_random_metadata_object()

        return Feature(shape=shape, dtype=dtype, metadata=metadata)

    return __make_random_feature


@pytest.fixture
def make_random_features_dict(make_random_valid_identifier, make_random_feature):
    def __make_random_features_dict():
        num_features = random.randint(1, 5)
        features = OrderedDict(
            [
                (make_random_valid_identifier(), make_random_feature())
                for _ in range(num_features)
            ]
        )
        return features

    return __make_random_features_dict


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
        for feat_name, feat in features.items():
            item_dict[feat_name] = make_random_data(feat.shape, feat.dtype)
        return item_dict

    return __make_random_item


@pytest.fixture
def make_random_item_dicts(make_random_item_dict):
    def __make_random_item_dicts(features):
        num_items = random.randint(1, 20)
        return [make_random_item_dict(features) for _ in range(num_items)]

    return __make_random_item_dicts


@pytest.fixture
def random_dataset_class(random_default_metadata, make_random_features_dict):
    namespec = dict()
    namespec.update(make_random_features_dict())
    namespec.update(DefaultMetadata=random_default_metadata)

    return type("RandomTestDataset", (Dataset,), namespec)


@pytest.fixture
def random_dataset_initialized(tmp_dataset_path, random_dataset_class):
    return random_dataset_class.initialize(tmp_dataset_path)


@pytest.fixture
def random_dataset_with_random_data(random_dataset_initialized, make_random_item_dict):
    dataset = random_dataset_initialized
    num_items = random.randint(1, 20)
    for _ in range(num_items):
        dataset.add_item(**make_random_item_dict(dataset.features))

    return dataset
