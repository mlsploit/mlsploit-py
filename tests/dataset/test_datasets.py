# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from mlsploit.dataset.datasets import *

# pylint: disable=unused-import
from .base.conftest import make_random_data, make_random_item_dict, tmp_dataset_path


def test_image_dataset(tmp_dataset_path):
    assert ImageDataset.metadata["module"] == ""
    assert "filename" in ImageDataset.features
    assert "image" in ImageDataset.features

    dataset = ImageDataset.initialize(tmp_dataset_path, module="testmodule")
    assert dataset.metadata["module"] == "testmodule"


def test_image_prediction_dataset():
    assert ImageClassificationDatasetWithPrediction.metadata["module"] == ""
    assert "filename" in ImageClassificationDatasetWithPrediction.features
    assert "image" in ImageClassificationDatasetWithPrediction.features
    assert "label" in ImageClassificationDatasetWithPrediction.features
    assert "prediction" in ImageClassificationDatasetWithPrediction.features


def test_cross_compatibility(tmp_dataset_path, make_random_item_dict):
    dataset = ImageClassificationDataset.initialize(
        tmp_dataset_path, module="testmodule"
    )

    items = [make_random_item_dict(dataset.features) for _ in range(10)]
    for item in items:
        dataset.add_item(**item)

    test_dataset = ImageDataset(tmp_dataset_path)
    assert dataset.metadata["module"] == "testmodule"
    for i, item in enumerate(test_dataset):
        assert item.filename == items[i]["filename"]
        assert np.all(item.image == items[i]["image"])

    with pytest.raises(RuntimeError) as excinfo:
        ImageClassificationDatasetWithPrediction(tmp_dataset_path)
    assert "incompatible" in str(excinfo)
