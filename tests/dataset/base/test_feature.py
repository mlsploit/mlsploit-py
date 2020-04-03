import numpy as np
import pytest

from mlsploit.dataset import Feature, Metadata
from mlsploit.dataset.base.feature import FeatureMap


def test_feature(make_random_metadata_object):
    metadata = make_random_metadata_object()
    vector_feat = Feature(shape=(123,), dtype=int, metadata=metadata)
    assert vector_feat.shape == (123,)
    assert vector_feat.dtype == np.dtype(int)
    assert vector_feat.metadata == metadata


def test_feature_eq(make_random_feature):
    feat1 = make_random_feature()
    feat2 = Feature(**feat1.dict())
    assert feat1 is not feat2
    assert feat1 == feat2

    feat3 = make_random_feature()
    while feat3.dict() == feat1.dict():
        feat3 = make_random_feature()
    assert feat3 != feat1


def test_feature_serialize_deserialize(make_random_feature):
    feat = make_random_feature()
    feat_deserialized = Feature.deserialize(feat.serialize())
    assert feat_deserialized == feat


def test_feature_map(make_random_features_dict):
    random_features_dict = make_random_features_dict()
    features = FeatureMap(**random_features_dict)
    for (k_dt, v_dt), (k_fm, v_fm) in zip(
        random_features_dict.items(), features.items()
    ):
        assert k_fm == k_dt
        assert v_fm == v_dt


def test_feature_map_eq(make_random_features_dict):
    random_features_dict = make_random_features_dict()
    feats1 = FeatureMap(**random_features_dict)
    feats2 = FeatureMap(**random_features_dict)
    assert feats1 == feats2

    feats3 = FeatureMap(**make_random_features_dict())
    while list(feats3.keys()) == list(feats1.keys()):
        feats3 = FeatureMap(**make_random_features_dict())
    assert feats3 != feats1


def test_feature_map_serialize_deserialize(make_random_features_dict):
    random_features_dict = make_random_features_dict()
    features = FeatureMap(**random_features_dict)
    features_deserialize = FeatureMap.deserialize(features.serialize())
    assert features_deserialize == features
