from collections import OrderedDict
import json
from typing import Mapping, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field, validator

from .metadata import Metadata


__all__ = ["Feature"]


FeatureType = Union["Feature", Type["Feature"]]


class Feature(BaseModel):
    shape: Optional[Tuple[int, ...]] = Field(...)
    dtype: np.dtype = Field(...)
    metadata: Metadata = Field(default=Metadata())

    class Config:
        # pylint: disable=too-few-public-methods
        allow_mutation = False
        arbitrary_types_allowed = True

    def __str__(self):
        return super().__repr__()

    def __eq__(self, o):
        return self.serialize() == o.serialize()

    @validator("dtype", pre=True)
    def _cast_to_np_dtype(cls, v):
        # pylint: disable=no-self-argument,no-self-use
        return np.dtype(v)

    def serialize(self):
        data = self.dict()
        data["shape"] = list(data["shape"]) if data["shape"] is not None else None
        data["dtype"] = self.dtype.name
        data["metadata"] = self.metadata.dict()
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data: str) -> "Feature":
        d = json.loads(data)
        d["shape"] = tuple(d["shape"]) if d["shape"] is not None else None
        d["dtype"] = np.dtype(d["dtype"])
        d["metadata"] = Metadata(**d["metadata"])
        return cls(**d)


class FeatureMap(OrderedDict, Mapping[str, FeatureType]):
    def __repr__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(["%s=%s" % it for it in self.items()]),
        )

    def __eq__(self, o):
        return self.serialize() == o.serialize()

    def serialize(self):
        return json.dumps(
            [[feat_name, feat.serialize()] for feat_name, feat in self.items()]
        )

    @classmethod
    def deserialize(cls, data: str) -> "FeatureMap":
        feats = list()
        for feat_name, feat_serialized in json.loads(data):
            feats.append((feat_name, Feature.deserialize(feat_serialized)))
        return cls(feats)
