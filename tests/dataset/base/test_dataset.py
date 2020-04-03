import numpy as np
import pytest

from mlsploit.dataset import Dataset, Feature, Metadata
from mlsploit.dataset.base.feature import FeatureMap

from .constants import RECOMMENDED_FILENAME


def test_dataset_recommended_filename():
    assert Dataset.recommended_filename == RECOMMENDED_FILENAME

    with pytest.raises(AttributeError) as excinfo:
        Dataset.recommended_filename = "something"
    assert "can't set attribute" in str(excinfo)


def test_dataset_subclass():
    class TestDataset(Dataset):
        feat1 = Feature(shape=(1, 2, 3), dtype=int)
        feat2 = Feature(
            shape=None, dtype=str, metadata=Metadata(abc=123, xyz=False, pi=3.14)
        )

        class DefaultMetadata:
            attr1 = "val1"
            attr2 = "val2"

    assert TestDataset.features["feat1"] == Feature(shape=(1, 2, 3), dtype=int)
    assert TestDataset.features["feat2"] == Feature(
        shape=None, dtype=str, metadata=Metadata(abc=123, xyz=False, pi=3.14)
    )

    assert TestDataset.metadata["attr1"] == "val1"
    assert TestDataset.metadata["attr2"] == "val2"


def test_dataset_initialize(tmp_dataset_path, random_dataset_class):
    assert not tmp_dataset_path.exists()
    dataset = random_dataset_class.initialize(tmp_dataset_path, testmetadata="testval")
    assert tmp_dataset_path.exists()
    assert dataset.metadata["testmetadata"] == "testval"


def test_dataset_read_metadata(tmp_dataset_path, random_dataset_class):
    dataset = random_dataset_class.initialize(tmp_dataset_path)
    assert isinstance(dataset.metadata, Metadata)
    assert random_dataset_class.read_metadata(tmp_dataset_path) == dataset.metadata


def test_dataset_read_features(tmp_dataset_path, random_dataset_class):
    dataset = random_dataset_class.initialize(tmp_dataset_path)
    assert isinstance(dataset.features, FeatureMap)
    assert random_dataset_class.read_features(tmp_dataset_path) == dataset.features

    assert Dataset.read_features(tmp_dataset_path) == dataset.features


def test_dataset_add_item(random_dataset_initialized, make_random_item_dicts):
    dataset = random_dataset_initialized

    num_items_added = 0
    assert len(dataset) == num_items_added

    item_dicts = make_random_item_dicts(dataset.features)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)
        num_items_added += 1
        assert len(dataset) == num_items_added


def test_dataset_get_item(random_dataset_initialized, make_random_item_dicts):
    dataset = random_dataset_initialized

    item_dicts = make_random_item_dicts(dataset.features)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)

    for i, item_dict in enumerate(item_dicts):
        item = dataset[i]
        for k, v in item_dict.items():
            assert np.all(getattr(item, k) == v)


def test_dataset_iter(random_dataset_initialized, make_random_item_dicts):
    dataset = random_dataset_initialized

    item_dicts = make_random_item_dicts(dataset.features)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)

    for i, item in enumerate(dataset):
        item_dict = item_dicts[i]
        for k, v in item_dict.items():
            assert np.all(getattr(item, k) == v)


def test_dataset_load(random_dataset_with_random_data):
    base_dataset = random_dataset_with_random_data
    loaded_dataset = Dataset(base_dataset.path)

    assert loaded_dataset.path == base_dataset.path
    assert loaded_dataset.metadata == base_dataset.metadata
    assert loaded_dataset.features == base_dataset.features

    num_items_checked = 0
    for base_item, loaded_item in zip(base_dataset, loaded_dataset):
        for feat_name, feat in loaded_dataset.features.items():
            base_val = getattr(base_item, feat_name)
            loaded_val = getattr(loaded_item, feat_name)
            assert np.all(base_val == loaded_val)
        num_items_checked += 1

    assert len(loaded_dataset) == len(base_dataset) == num_items_checked
