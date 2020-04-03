from collections import OrderedDict
import random

import pytest

from mlsploit.dataset import Metadata


def test_metadata(make_random_metadata_dict):
    random_metadata_dict = make_random_metadata_dict()
    metadata = Metadata(**random_metadata_dict)
    for k, v in random_metadata_dict.items():
        assert metadata[k] == v


def test_metadata_empty():
    metadata = Metadata()
    assert len(metadata) == 0
    assert metadata.serialize() == "[]"


def test_metadata_immutable(make_random_metadata_object):
    random_metadata_object = make_random_metadata_object()
    assert isinstance(random_metadata_object, Metadata)
    with pytest.raises(TypeError) as excinfo:
        random_metadata_object["abc"] = 123
    assert "does not support item assignment" in str(excinfo)


def test_metadata_eq(make_random_metadata_dict):
    items = sorted(make_random_metadata_dict().items())
    md1 = Metadata(**OrderedDict(items))
    md2 = Metadata(**OrderedDict(items))
    assert md1 == md2

    items1 = sorted(make_random_metadata_dict().items())
    items2 = sorted(make_random_metadata_dict().items())
    while items1 == items2:
        items2 = sorted(make_random_metadata_dict().items())

    md1 = Metadata(**OrderedDict(items1))
    md2 = Metadata(**OrderedDict(items2))
    assert md1 != md2


def test_metadata_len(make_random_metadata_dict):
    random_metadata_dict = make_random_metadata_dict()
    metadata = Metadata(**random_metadata_dict)
    assert len(metadata) == len(random_metadata_dict)


def test_metadata_iter(make_random_metadata_dict):
    random_metadata_dict = make_random_metadata_dict()
    metadata = Metadata(**random_metadata_dict)
    for k in metadata:
        assert k in random_metadata_dict


def test_metadata_contains(make_random_metadata_dict):
    random_metadata_dict = make_random_metadata_dict()
    metadata = Metadata(**random_metadata_dict)
    for k in random_metadata_dict:
        assert k in metadata
        assert (k + "abc") not in metadata


def test_metadata_dict(make_random_metadata_dict):
    random_metadata_dict = make_random_metadata_dict()
    items = sorted(random_metadata_dict.items())
    metadata = Metadata(**OrderedDict(items))

    for (k_it, v_it), k_md in zip(items, metadata.keys()):
        assert k_md == k_it

    for (k_it, v_it), v_md in zip(items, metadata.values()):
        assert v_md == v_it

    for (k_it, v_it), (k_md, v_md) in zip(items, metadata.items()):
        assert k_md == k_it
        assert v_md == v_it


def test_metadata_serialize_deserialize(make_random_metadata_object):
    random_metadata_object = make_random_metadata_object()
    deserialized_metadata = Metadata.deserialize(random_metadata_object.serialize())
    assert deserialized_metadata == random_metadata_object
