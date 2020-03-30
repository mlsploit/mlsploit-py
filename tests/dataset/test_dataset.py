from devtools import debug
import numpy as np
from pydantic.error_wrappers import ValidationError
import pytest

from mlsploit.dataset import Dataset
from mlsploit.dataset.base import Feature

from .constants import *


def test_dataset_metaclass():
    assert Dataset.recommended_filename == RECOMMENDED_FILENAME

    with pytest.raises(AttributeError) as excinfo:
        Dataset.recommended_filename = "something"
    assert "can't set attribute" in str(excinfo)


def test_feature_init():
    feat = Feature(name="vector", shape=(123,), dtype=int)
    assert feat.name == "vector"
    assert feat.shape == (123,)
    assert feat.dtype is np.dtype(int)


def test_feature_immutable():
    feat = Feature(name="tensor", shape=(123, 456, 789), dtype=int)
    assert feat.name == "tensor"
    assert feat.shape == (123, 456, 789)
    assert feat.dtype is np.dtype(int)

    with pytest.raises(TypeError) as excinfo:
        feat.dtype = str
    assert '"Feature" is immutable' in str(excinfo)


def test_feature_serialize_deserialize(random_features):
    for feature in random_features:
        feature_serialized = feature.serialize()

        assert (
            feature_serialized["shape"] is None
            or type(feature_serialized["shape"]) is list
        )
        assert "dtype" not in feature_serialized
        assert "dtype_name" in feature_serialized
        assert type(feature_serialized["dtype_name"]) is str

        assert Feature.deserialize(feature_serialized) == feature


def test_dataset_build_with_features(tmp_dataset_path, random_features):
    assert not tmp_dataset_path.exists()

    dataset = Dataset.build(tmp_dataset_path)
    for feature in random_features:
        feat_name, feat_shape, feat_dtype = (
            feature.name,
            feature.shape,
            feature.dtype,
        )
        dataset.add_feature(name=feat_name, shape=feat_shape, dtype=feat_dtype)

    assert not tmp_dataset_path.exists()
    dataset.conclude_build()
    assert tmp_dataset_path.exists()
    assert dataset.path == tmp_dataset_path.resolve()

    features = dataset.features
    assert all(features[i] == feature for i, feature in enumerate(random_features))
    assert len(features) == len(random_features)


def test_feature_name_not_identifier(tmp_dataset_path, make_random_invalid_identifier):

    for _ in range(100):
        dataset = Dataset.build(tmp_dataset_path)
        invalid_name = make_random_invalid_identifier()
        with pytest.raises(ValueError) as excinfo:
            dataset.add_feature(name=invalid_name, shape=None, dtype=int)
        assert "has to be a valid python identifier" in str(excinfo)


def test_dataset_init_with_metadata(
    tmp_dataset_path, random_metadata_dict, random_features
):

    dataset = Dataset.build(tmp_dataset_path)
    for feature in random_features:
        dataset.add_feature(**feature.dict())
    for k, v in random_metadata_dict.items():
        dataset.with_metadata(**{k: v})

    dataset.conclude_build()

    for k, v in random_metadata_dict.items():
        assert getattr(dataset.metadata, k) == v


def test_dataset_read_metadata(random_empty_dataset):
    loaded_metadata = Dataset.read_metadata(random_empty_dataset.path)
    for k, v in random_empty_dataset.metadata._asdict().items():
        assert getattr(loaded_metadata, k) == v


def test_dataset_read_features(random_empty_dataset):
    loaded_features = Dataset.read_features(random_empty_dataset.path)
    assert isinstance(loaded_features, tuple)
    assert loaded_features == random_empty_dataset.features


def test_dataset_add_item(random_empty_dataset, make_random_item_dicts):
    dataset = random_empty_dataset

    num_items_added = 0
    assert len(dataset) == num_items_added

    item_dicts = make_random_item_dicts(dataset.features)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)
        num_items_added += 1
        assert len(dataset) == num_items_added


def test_dataset_get_item(random_empty_dataset, make_random_item_dicts):
    dataset = random_empty_dataset
    item_dicts = make_random_item_dicts(dataset.features)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)

    for i, item_dict in enumerate(item_dicts):
        item = dataset[i]
        for k, v in item_dict.items():
            assert np.all(getattr(item, k) == v)


def test_dataset_iter(random_empty_dataset, make_random_item_dicts):
    dataset = random_empty_dataset
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
    assert loaded_dataset.features == base_dataset.features
    for k, v in base_dataset.metadata._asdict().items():
        assert getattr(loaded_dataset.metadata, k) == v

    num_items_checked = 0
    for base_item, loaded_item in zip(base_dataset, loaded_dataset):
        for feature in loaded_dataset.features:
            feat_name = feature.name

            base_val = getattr(base_item, feat_name)
            loaded_val = getattr(loaded_item, feat_name)
            assert np.all(base_val == loaded_val), (
                np.mean(base_val),
                np.mean(loaded_val),
            )

        num_items_checked += 1

    assert len(loaded_dataset) == len(base_dataset) == num_items_checked


def test_dataset_build_mode(
    tmp_dataset_path, make_random_feature, make_random_item_dict
):
    dataset = Dataset.build(tmp_dataset_path)

    with pytest.raises(AttributeError):
        _ = dataset.metadata

    with pytest.raises(AttributeError):
        _ = dataset.features

    with pytest.raises(RuntimeError) as excinfo:
        len(dataset)
    assert "disabled in Dataset build mode" in str(excinfo)

    with pytest.raises(RuntimeError) as excinfo:
        _ = dataset[0]
    assert "disabled in Dataset build mode" in str(excinfo)

    with pytest.raises(RuntimeError) as excinfo:
        for _ in dataset:
            pass
    assert "disabled in Dataset build mode" in str(excinfo)

    with pytest.raises(RuntimeError) as excinfo:
        dataset.add_item()
    assert "disabled in Dataset build mode" in str(excinfo)

    feature = make_random_feature()
    dataset.add_feature(**feature.dict())
    dataset = dataset.conclude_build()

    dataset.add_item(**make_random_item_dict([feature]))

    num_iterated = 0
    for _ in dataset:
        num_iterated += 1
    assert num_iterated == 1
    assert len(dataset) == 1
    assert len(dataset.features) == 1
    assert dataset.features[0] == feature
    assert dataset.metadata._asdict() == {}

    with pytest.raises(RuntimeError) as excinfo:
        dataset.with_metadata()
    assert "is only enabled in Dataset build mode" in str(excinfo)

    with pytest.raises(RuntimeError) as excinfo:
        dataset.add_feature(**make_random_feature().dict())
    assert "is only enabled in Dataset build mode" in str(excinfo)

    with pytest.raises(RuntimeError) as excinfo:
        dataset.conclude_build()
    assert "is only enabled in Dataset build mode" in str(excinfo)
