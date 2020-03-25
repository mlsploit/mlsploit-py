from devtools import debug
import numpy as np
from pydantic.error_wrappers import ValidationError
import pytest

from mlsploit.dataset import Dataset
from mlsploit.dataset.base import ItemAttr

from .constants import *


def test_dataset_metaclass():
    assert Dataset.recommended_filename == RECOMMENDED_FILENAME

    with pytest.raises(AttributeError) as excinfo:
        Dataset.recommended_filename = "something"
    assert "can't set attribute" in str(excinfo)


def test_dataset_item_attr_init():
    attr = ItemAttr(name="vector", shape=(123,), dtype=int)
    assert attr.name == "vector"
    assert attr.shape == (123,)
    assert attr.dtype is np.dtype(int)


def test_dataset_item_attr_immutable():
    attr = ItemAttr(name="tensor", shape=(123, 456, 789), dtype=int)
    assert attr.name == "tensor"
    assert attr.shape == (123, 456, 789)
    assert attr.dtype is np.dtype(int)

    with pytest.raises(TypeError) as excinfo:
        attr.dtype = str
    assert '"ItemAttr" is immutable' in str(excinfo)


def test_dataset_item_attr_serialize_deserialize(random_item_attrs):
    for item_attr in random_item_attrs:
        item_attr_serialized = item_attr.serialize()

        assert (
            item_attr_serialized["shape"] is None
            or type(item_attr_serialized["shape"]) is list
        )
        assert "dtype" not in item_attr_serialized
        assert "dtype_name" in item_attr_serialized
        assert type(item_attr_serialized["dtype_name"]) is str

        assert ItemAttr.deserialize(item_attr_serialized) == item_attr


def test_dataset_build_with_item_attrs(tmp_dataset_path, random_item_attrs):
    assert not tmp_dataset_path.exists()

    dataset_builder = Dataset.build(tmp_dataset_path)
    for item_attr in random_item_attrs:
        attr_name, attr_shape, attr_dtype = (
            item_attr.name,
            item_attr.shape,
            item_attr.dtype,
        )

        dataset_builder.add_item_attr(
            name=attr_name, shape=attr_shape, dtype=attr_dtype
        )
    dataset = dataset_builder.conclude_build()

    assert tmp_dataset_path.exists()
    assert dataset.path == tmp_dataset_path.resolve()

    item_attrs = dataset.item_attrs
    assert all(
        item_attrs[i] == item_attr for i, item_attr in enumerate(random_item_attrs)
    )
    assert len(item_attrs) == len(random_item_attrs)


def test_dataset_item_attr_name_not_identifier(
    tmp_dataset_path, make_random_invalid_identifier
):

    for _ in range(100):
        dataset_builder = Dataset.build(tmp_dataset_path)
        invalid_name = make_random_invalid_identifier()
        with pytest.raises(ValueError) as excinfo:
            dataset_builder.add_item_attr(name=invalid_name, shape=None, dtype=int)
        assert "has to be a valid python identifier" in str(excinfo)


def test_dataset_init_with_metadata(
    tmp_dataset_path, random_metadata_dict, random_item_attrs
):

    dataset_builder = Dataset.build(tmp_dataset_path)
    for item_attr in random_item_attrs:
        dataset_builder.add_item_attr(**item_attr.dict())
    for k, v in random_metadata_dict.items():
        dataset_builder.with_metadata(**{k: v})

    dataset = dataset_builder.conclude_build()

    for k, v in random_metadata_dict.items():
        assert getattr(dataset.metadata, k) == v


def test_dataset_read_metadata(random_empty_dataset):
    loaded_metadata = Dataset.read_metadata(random_empty_dataset.path)
    for k, v in random_empty_dataset.metadata._asdict().items():
        assert getattr(loaded_metadata, k) == v


def test_dataset_read_item_attrs(random_empty_dataset):
    loaded_item_attrs = Dataset.read_item_attrs(random_empty_dataset.path)
    assert isinstance(loaded_item_attrs, tuple)
    assert loaded_item_attrs == random_empty_dataset.item_attrs


def test_dataset_add_item(random_empty_dataset, make_random_item_dicts):
    dataset = random_empty_dataset

    num_items_added = 0
    assert len(dataset) == num_items_added

    item_dicts = make_random_item_dicts(dataset.item_attrs)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)
        num_items_added += 1
        assert len(dataset) == num_items_added


def test_dataset_get_item(random_empty_dataset, make_random_item_dicts):
    dataset = random_empty_dataset
    item_dicts = make_random_item_dicts(dataset.item_attrs)
    for item_dict in item_dicts:
        dataset.add_item(**item_dict)

    for i, item_dict in enumerate(item_dicts):
        item = dataset[i]
        for k, v in item_dict.items():
            assert np.all(getattr(item, k) == v)


def test_dataset_iter(random_empty_dataset, make_random_item_dicts):
    dataset = random_empty_dataset
    item_dicts = make_random_item_dicts(dataset.item_attrs)
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
    assert loaded_dataset.item_attrs == base_dataset.item_attrs
    for k, v in base_dataset.metadata._asdict().items():
        assert getattr(loaded_dataset.metadata, k) == v

    num_items_checked = 0
    for base_item, loaded_item in zip(base_dataset, loaded_dataset):
        for item_attr in loaded_dataset.item_attrs:
            attr_name = item_attr.name

            base_val = getattr(base_item, attr_name)
            loaded_val = getattr(loaded_item, attr_name)
            assert np.all(base_val == loaded_val), (
                np.mean(base_val),
                np.mean(loaded_val),
            )

        num_items_checked += 1

    assert len(loaded_dataset) == len(base_dataset) == num_items_checked
