from collections import namedtuple, OrderedDict
from functools import partial
import inspect
from pathlib import Path
from typing import Any, Generator, Tuple
import warnings

from filelock import FileLock
import numpy as np
import zarr

from ...paths import FilepathType
from .feature import Feature, FeatureMap
from .metadata import Metadata, PrimitiveType


__all__ = ["Dataset"]


_METADATA_KEY = "__METADATA__"
_FEATURES_KEY = "__FEATURES__"
_RECOMMENDED_FILENAME = "MLSPLOIT.dset.zip"

DynamicNamedTuple = Tuple[Any, ...]

# pylint: disable=invalid-name
ItemViewMeta = partial(namedtuple, "ItemView")
ItemSetViewMeta = partial(namedtuple, "ItemSetView")
# pylint: enable=invalid-name


def _lockfile(path: FilepathType):
    path = Path(path).resolve()
    filedir = path.parent
    filename = path.name
    return filedir / (filename + ".lock")


def _build_dataset(path: FilepathType, metadata: Metadata, features: FeatureMap):
    path = Path(path).resolve()
    lock = FileLock(_lockfile(path))
    with lock, zarr.ZipStore(path, mode="w") as store:
        root = zarr.group(store=store)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            root.attrs[_METADATA_KEY] = metadata.serialize()
            root.attrs[_FEATURES_KEY] = features.serialize()

        for feat_name, feat in features.items():
            feat_dtype = feat.dtype if feat.dtype.name != "str" else str
            collection_shape = (0,) + feat.shape if feat.shape is not None else (0,)
            chunk_shape = (1,) + feat.shape if feat.shape is not None else (1,)

            root.create_dataset(
                feat_name, shape=collection_shape, chunks=chunk_shape, dtype=feat_dtype
            )


def _add_item(path: FilepathType, features: FeatureMap, input_dict: dict):
    parsed_input_dict = dict()
    for feat_name, feat in features.items():
        try:
            feat_val = input_dict[feat_name]
        except KeyError:
            raise RuntimeError(f"Feature missing ({feat_name})")

        if feat.shape is not None and np.array(feat_val).shape != feat.shape:
            raise ValueError(
                f"Shape mismatch for {feat_name} "
                f"(expected {feat.shape}, "
                f"got {np.array(feat_val).shape})"
            )

        feat_val = (
            np.expand_dims(feat_val, axis=0)
            if isinstance(feat_val, np.ndarray)
            else np.array([feat_val], dtype=feat.dtype)
        ).astype(feat.dtype)

        parsed_input_dict[feat_name] = feat_val

    path = Path(path).resolve()
    lock = FileLock(_lockfile(path))
    with lock, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        store = zarr.ZipStore(path, mode="a")
        root = zarr.group(store=store)
        for k, v in parsed_input_dict.items():
            root[k].append(v)
        store.close()


class _DatasetMeta(type):
    def __new__(cls, name, bases, nmspc):
        metadata = OrderedDict()
        features = OrderedDict()

        for base in reversed(bases):
            if hasattr(base, "metadata") and isinstance(base.metadata, Metadata):
                metadata.update(base.metadata.dict())

            if hasattr(base, "features") and isinstance(base.features, FeatureMap):
                features.update(base.features)

        if "DefaultMetadata" in nmspc:
            for k, v in inspect.getmembers(nmspc["DefaultMetadata"]):
                if (not inspect.isroutine(v)) and (
                    not (k.startswith("__") and k.endswith("__"))
                ):
                    metadata[k] = v

        for k, v in nmspc.items():
            if isinstance(v, Feature):
                features[k] = v

        metadata = Metadata(**metadata)
        features = FeatureMap(**features)

        nmspc.update(metadata=metadata, features=features)

        return super().__new__(cls, name, bases, nmspc)

    @property
    def recommended_filename(cls):
        return _RECOMMENDED_FILENAME


class Dataset(metaclass=_DatasetMeta):
    metadata: Metadata
    features: FeatureMap

    class DefaultMetadata:
        # pylint: disable=too-few-public-methods
        pass

    def __init__(self, path: FilepathType):
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        self.path = path
        self.metadata = self.read_metadata(path)
        self.features = self.read_features(path)

        _ = len(self)  # health check

    def __repr__(self):
        return "%s(path=%s, metadata=%s, features=%s, len=%d)" % (
            self.__class__.__name__,
            self.path,
            self.metadata,
            self.features,
            len(self),
        )

    def __len__(self):
        sizes = set()
        lock = FileLock(_lockfile(self.path))
        with lock, zarr.ZipStore(self.path, mode="r") as store:
            root = zarr.group(store=store)
            for feat_name in self.features.keys():
                sizes.add(len(root[feat_name]))
        if len(sizes) != 1:
            raise RuntimeError("Dataset corrupted!!!")
        return sizes.pop()

    def __getitem__(self, idx) -> DynamicNamedTuple:
        feat_names = list(self.features.keys())
        lock = FileLock(_lockfile(self.path))
        with lock, zarr.ZipStore(self.path, mode="r") as store:
            root = zarr.group(store=store)
            data_dict = {feat_name: root[feat_name][idx] for feat_name in feat_names}

        return (
            ItemViewMeta(feat_names)(**data_dict)
            if isinstance(idx, int)
            else ItemSetViewMeta(feat_names)(**data_dict)
        )

    def __iter__(self) -> Generator[DynamicNamedTuple, None, None]:
        for i in range(len(self)):
            yield self[i]

    def add_item(self, **feature_kwargs):
        initial_size = len(self)

        _add_item(self.path, self.features, feature_kwargs)

        if len(self) != initial_size + 1:
            raise RuntimeError(
                "Dataset got corrupted while trying to add %s" % feature_kwargs
            )

    @classmethod
    def info(cls, print_: bool = True) -> str:
        info_ = "%s(metadata=%s, features=%s)" % (
            cls.__name__,
            cls.metadata,
            cls.features,
        )

        if print_:
            print(info_)

        return info_

    @classmethod
    def initialize(cls, path: FilepathType, **metadata_kwargs: PrimitiveType):
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(path)

        metadata_dict = cls.metadata.dict()
        metadata_dict.update(metadata_kwargs)

        metadata = Metadata(**metadata_dict)
        features = cls.features
        _build_dataset(path, metadata, features)

        return cls(path)

    @classmethod
    def read_metadata(cls, path: FilepathType) -> Metadata:
        path = Path(path).resolve()
        lock = FileLock(_lockfile(path))
        with lock, zarr.ZipStore(path, mode="r") as store:
            root = zarr.group(store=store)
            metadata = root.attrs[_METADATA_KEY]
        return Metadata.deserialize(metadata)

    @classmethod
    def read_features(cls, path: FilepathType) -> FeatureMap:
        path = Path(path).resolve()
        lock = FileLock(_lockfile(path))
        with lock, zarr.ZipStore(path, mode="r") as store:
            root = zarr.group(store=store)
            features = root.attrs[_FEATURES_KEY]
        raw_features = FeatureMap.deserialize(features)

        if len(cls.features) == 0:
            return raw_features

        for required_feat_name, required_feat in cls.features.items():
            if (
                required_feat_name not in raw_features
                or raw_features[required_feat_name] != required_feat
            ):
                raise RuntimeError(
                    "Dataset stored at %s is incompatible with %s"
                    % (path, cls.__class__)
                )

        return cls.features
