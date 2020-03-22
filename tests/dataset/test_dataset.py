from devtools import debug
import numpy as np
from pydantic.error_wrappers import ValidationError
import pytest

from mlsploit.dataset import Dataset

from .constants import *


def test_dataset_metaclass():
    assert Dataset.recommended_filename == RECOMMENDED_FILENAME

    with pytest.raises(AttributeError) as excinfo:
        Dataset.recommended_filename = 'something'
    assert 'can\'t set attribute' in str(excinfo)


def test_dataset_item_attr_init():
    attr = Dataset.ItemAttr(
        name='vector',
        shape=(123,),
        dtype=int)
    assert attr.name == 'vector'
    assert attr.shape == (123,)
    assert attr.dtype is int


def test_dataset_item_attr_immutable():
    attr = Dataset.ItemAttr(
        name='tensor',
        shape=(123, 456, 789),
        dtype=int)
    assert attr.name == 'tensor'
    assert attr.shape == (123, 456, 789)
    assert attr.dtype is int

    with pytest.raises(TypeError) as excinfo:
        attr.dtype = str
    assert '"ItemAttr" is immutable' in str(excinfo)


@pytest.mark.parametrize('attr_name', [ATTRS_KEY, METADATA_KEY])
def test_dataset_item_attr_reserved_name(attr_name):
    with pytest.raises(ValueError) as excinfo:
        Dataset.ItemAttr(
            name=attr_name,
            shape=(123,),
            dtype=int)
    assert ('Item attribute name cannot be %s'
            % attr_name) in str(excinfo)


def test_dataset_init_with_item_attrs(tmp_dataset_path, random_item_attrs):
    assert not tmp_dataset_path.exists()

    dataset = Dataset(
        tmp_dataset_path,
        random_item_attrs)

    assert tmp_dataset_path.exists()
    assert dataset.path == tmp_dataset_path.resolve()

    item_attrs = dataset.item_attrs
    assert all(item_attrs[i] == item_attr
               for i, item_attr in enumerate(
                   random_item_attrs))
    assert len(item_attrs) == len(random_item_attrs)


def test_dataset_init_with_one_item_attr(tmp_dataset_path, make_random_item_attr):

    item_attr = make_random_item_attr()
    dataset = Dataset(tmp_dataset_path, item_attr)

    assert len(dataset.item_attrs) == 1
    assert dataset.item_attrs[0] == item_attr


def test_dataset_item_attr_name_not_identifier(
        tmp_dataset_path, make_random_invalid_identifier):

    for _ in range(100):
        invalid_name = make_random_invalid_identifier()
        with pytest.raises(ValueError) as excinfo:
            Dataset(tmp_dataset_path,
                    Dataset.ItemAttr(
                        name=invalid_name,
                        shape=None, dtype=int))
        assert 'must be valid identifiers' in str(excinfo)


def test_dataset_init_with_metadata(
        tmp_dataset_path, random_metadata_dict,
        random_item_attrs):

    dataset = Dataset(
        tmp_dataset_path,
        random_item_attrs,
        metadata=random_metadata_dict)

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
    loaded_dataset = Dataset.load(base_dataset.path)

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
            assert np.all(base_val == loaded_val), (np.mean(base_val), np.mean(loaded_val))

        num_items_checked += 1

    assert len(loaded_dataset) == len(base_dataset) == num_items_checked
