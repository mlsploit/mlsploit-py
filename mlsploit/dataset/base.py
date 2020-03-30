from collections import namedtuple
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generator, Optional, Tuple, Type, Union
import warnings

import numpy as np
from pydantic import BaseModel, Field, validator
import zarr

from ..paths import FilepathType


__all__ = ["Dataset"]


DynamicNamedTuple = Tuple[Any, ...]

_METADATA_KEY = "__METADATA__"
_FEATURES_KEY = "__FEATURES__"
_RECOMMENDED_FILENAME = "MLSPLOIT.dset"

# pylint: disable=invalid-name
ItemViewMeta = partial(namedtuple, "ItemView")
ItemSetViewMeta = partial(namedtuple, "ItemSetView")
MetadataViewMeta = partial(namedtuple, "MetadataView")
FeatureSetViewMeta = partial(namedtuple, "FeatureSetView")
# pylint: enable=invalid-name


def enable_only_in_build_mode(instance_method):
    def wrapper(self, *args, **kwargs):
        # pylint: disable=protected-access
        if not self._build_mode:
            raise RuntimeError(
                "%s is only enabled in Dataset build mode" % instance_method.__name__
            )

        return instance_method(self, *args, **kwargs)

    return wrapper


def disable_in_build_mode(instance_method):
    def wrapper(self, *args, **kwargs):
        # pylint: disable=protected-access
        if self._build_mode:
            raise RuntimeError(
                "%s is disabled in Dataset build mode" % instance_method.__name__
            )

        return instance_method(self, *args, **kwargs)

    return wrapper


class Feature(BaseModel):
    name: str
    shape: Optional[Tuple[int, ...]] = Field(...)
    dtype: np.dtype

    class Config:
        # pylint: disable=too-few-public-methods
        allow_mutation = False
        arbitrary_types_allowed = True

    @validator("name")
    def _ensure_name_is_identifier(cls, v):
        # pylint: disable=no-self-argument,no-self-use
        if not v.isidentifier():
            raise ValueError(
                "Feature name has to be " "a valid python identifier (got: %s)" % v
            )
        return v

    @validator("dtype", pre=True)
    def _cast_to_np_dtype(cls, v):
        # pylint: disable=no-self-argument,no-self-use
        return np.dtype(v)

    def serialize(self):
        data = self.dict()
        data["shape"] = list(data["shape"]) if data["shape"] is not None else None
        data["dtype_name"] = self.dtype.name
        del data["dtype"]
        return data

    @classmethod
    def deserialize(cls, data):
        return cls(name=data["name"], shape=data["shape"], dtype=data["dtype_name"])


class DatasetMeta(type):
    @property
    def recommended_filename(cls):
        return _RECOMMENDED_FILENAME


class DatasetProto:
    # pylint: disable=too-few-public-methods
    _build_mode: bool
    _lock: Lock

    path: FilepathType
    metadata: DynamicNamedTuple
    features: DynamicNamedTuple

    add_item: Callable[..., None]
    read_metadata: Callable[[FilepathType], DynamicNamedTuple]
    read_features: Callable[[FilepathType], DynamicNamedTuple]


class DatasetBuilderMixin(DatasetProto):
    @enable_only_in_build_mode
    def _init_build_mode(self):
        self._build_metadata = dict()
        self._build_features = list()

    @enable_only_in_build_mode
    def _setup_dataset(self):
        features_serialized = list(map(lambda f: f.serialize(), self._build_features))

        with self._lock, zarr.ZipStore(self.path, mode="w") as store:
            root = zarr.group(store=store)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)

                root.attrs[_METADATA_KEY] = dict(self._build_metadata)
                root.attrs[_FEATURES_KEY] = features_serialized

            for feature_serialized in features_serialized:
                feat_name = feature_serialized["name"]
                feat_dtype = feature_serialized["dtype_name"]
                feat_shape = (
                    tuple(feature_serialized["shape"])
                    if feature_serialized["shape"] is not None
                    else None
                )

                collection_shape = (0,) + feat_shape if feat_shape is not None else (0,)
                chunk_shape = (1,) + feat_shape if feat_shape is not None else (1,)

                root.create_dataset(
                    feat_name,
                    shape=collection_shape,
                    chunks=chunk_shape,
                    dtype=feat_dtype,
                )

    @enable_only_in_build_mode
    def with_metadata(self, **metadata_kwargs) -> Type[DatasetProto]:
        self._build_metadata.update(metadata_kwargs)
        return self

    @enable_only_in_build_mode
    def add_feature(
        self,
        name: str,
        shape: Union[Tuple[int, ...], None],
        dtype: Union[str, np.dtype],
    ) -> Type[DatasetProto]:

        self._build_features.append(Feature(name=name, shape=shape, dtype=dtype))
        return self

    @enable_only_in_build_mode
    def conclude_build(self) -> Type[DatasetProto]:
        self._setup_dataset()

        self._build_mode = False
        del self._build_metadata
        del self._build_features

        self.metadata = self.read_metadata(self.path)
        self.features = self.read_features(self.path)

        return self


class Dataset(DatasetBuilderMixin, metaclass=DatasetMeta):
    def __init__(self, filepath: FilepathType, build_mode: bool = False):
        filepath = Path(filepath).resolve()

        if build_mode and filepath.exists():
            raise FileExistsError(filepath)
        if (not build_mode) and (not filepath.exists()):
            raise FileNotFoundError(filepath)

        self._build_mode = build_mode
        self._lock = Lock()

        self.path = filepath
        if build_mode:
            self._init_build_mode()
        else:
            self.metadata = self.read_metadata(filepath)  # pylint: disable=no-member
            self.features = self.read_features(filepath)  # pylint: disable=no-member

    def __repr__(self) -> str:
        if self._build_mode:
            return "<%s: '%s', BUILD_MODE>" % (self.__class__.__name__, self.path)

        return "<%s: '%s', %s, %s, len=%d>" % (
            self.__class__.__name__,
            self.path,
            self.metadata,
            self.features,
            len(self),
        )

    @disable_in_build_mode
    def __len__(self) -> int:
        sizes = set()
        with self._lock, zarr.ZipStore(self.path, mode="r") as store:
            root = zarr.group(store=store)
            for feature in self.features:
                sizes.add(len(root[feature.name]))
        if len(sizes) != 1:
            raise RuntimeError("Dataset corrupted!!!")
        return sizes.pop()

    @disable_in_build_mode
    def __getitem__(self, idx) -> DynamicNamedTuple:
        feat_names = list(map(lambda a: a.name, self.features))
        with self._lock, zarr.ZipStore(self.path, mode="r") as store:
            root = zarr.group(store=store)
            data_dict = {feat_name: root[feat_name][idx] for feat_name in feat_names}

        return (
            ItemViewMeta(feat_names)(**data_dict)
            if isinstance(idx, int)
            else ItemSetViewMeta(feat_names)(**data_dict)
        )

    @disable_in_build_mode
    def __iter__(self) -> Generator[DynamicNamedTuple, None, None]:
        for i in range(len(self)):
            yield self[i]

    @disable_in_build_mode
    def add_item(self, **feat_kwargs):
        initial_size = len(self)

        item_dict = dict()
        for feature in self.features:
            feat_name = feature.name
            feat_shape = feature.shape
            feat_dtype = feature.dtype.name

            try:
                feat_val = feat_kwargs[feat_name]
            except KeyError:
                raise RuntimeError(f"Feature missing: {feat_name}")

            if feat_shape is not None and np.array(feat_val).shape != feat_shape:
                raise ValueError(
                    f"Shape mismatch for {feat_name} "
                    f"(expected: {feat_shape}, "
                    f"got: {np.array(feat_val).shape})"
                )

            feat_val = (
                np.expand_dims(feat_val, axis=0)
                if isinstance(feat_val, np.ndarray)
                else np.array([feat_val], dtype=feat_dtype)
            )
            feat_val = feat_val.astype(feat_dtype)

            item_dict[feat_name] = feat_val

        with self._lock, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            store = zarr.ZipStore(self.path, mode="a")
            root = zarr.group(store=store)
            for k, v in item_dict.items():
                root[k].append(v)
            store.close()

        if len(self) != initial_size + 1:
            raise RuntimeError(
                "Dataset got corrupted while trying to add %s" % item_dict
            )

    @staticmethod
    def read_metadata(filepath: FilepathType) -> DynamicNamedTuple:
        filepath = Path(filepath).resolve()
        with zarr.ZipStore(filepath, mode="r") as store:
            root = zarr.group(store=store)
            metadata = root.attrs[_METADATA_KEY]
        return MetadataViewMeta(sorted(metadata.keys()))(**metadata)

    @staticmethod
    def read_features(filepath: FilepathType) -> DynamicNamedTuple:
        filepath = Path(filepath).resolve()
        with zarr.ZipStore(filepath, mode="r") as store:
            root = zarr.group(store=store)
            features_serialized = root.attrs[_FEATURES_KEY]

        features = list(map(Feature.deserialize, features_serialized))
        return FeatureSetViewMeta(map(lambda a: a.name, features))(*features)

    @classmethod
    def build(cls, filepath: FilepathType) -> "Dataset":
        return cls(filepath, build_mode=True)
