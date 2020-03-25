from collections import namedtuple
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Generator, Optional, Tuple, Union
import warnings

import numpy as np
from pydantic import BaseModel, Field, validator
import zarr

from ..paths import FilepathType


__all__ = ["Dataset"]


_METADATA_KEY = "__METADATA__"
_ITEMATTRS_KEY = "__ITEMATTRS__"
_RECOMMENDED_FILENAME = "MLSPLOIT.db"

PrimitiveType = Union[str, int, float, bool]

# pylint: disable=invalid-name
ItemViewMeta = partial(namedtuple, "ItemView")
ItemSetViewMeta = partial(namedtuple, "ItemSetView")
MetadataViewMeta = partial(namedtuple, "MetadataView")
ItemAttrSetViewMeta = partial(namedtuple, "ItemAttrSetView")
# pylint: enable=invalid-name


class ItemAttr(BaseModel):
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
                "Item attribute name has to be "
                "a valid python identifier (got: %s)" % v
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


class Dataset(metaclass=DatasetMeta):
    def __init__(self, filepath: FilepathType):
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(
                f'"{filepath}" '
                f"(use Dataset.build instead, "
                f"if you're trying to build a dataset)"
            )

        self._lock = Lock()

        self.path = filepath
        self.item_attrs = self.read_item_attrs(filepath)
        self.metadata = self.read_metadata(filepath)

    def __repr__(self) -> str:
        return "<%s: '%s', %s, %s, len=%d>" % (
            self.__class__.__name__,
            self.path,
            self.metadata,
            self.item_attrs,
            len(self),
        )

    def __len__(self) -> int:
        sizes = set()
        with self._lock, zarr.ZipStore(self.path, mode="r") as store:
            root = zarr.group(store=store)
            for item_attr in self.item_attrs:
                sizes.add(len(root[item_attr.name]))
        if len(sizes) != 1:
            raise RuntimeError("Dataset corrupted!!!")
        return sizes.pop()

    def __getitem__(self, idx) -> Union["ItemView", "ItemSetView"]:
        attr_names = list(map(lambda a: a.name, self.item_attrs))
        with self._lock, zarr.ZipStore(self.path, mode="r") as store:
            root = zarr.group(store=store)
            data_dict = {attr_name: root[attr_name][idx] for attr_name in attr_names}

        return (
            ItemViewMeta(attr_names)(**data_dict)
            if isinstance(idx, int)
            else ItemSetViewMeta(attr_names)(**data_dict)
        )

    def __iter__(self) -> Generator["ItemView", None, None]:
        for i in range(len(self)):
            yield self[i]

    def add_item(self, **attr_kwargs):
        initial_size = len(self)

        item_dict = dict()
        for item_attr in self.item_attrs:
            attr_name = item_attr.name
            attr_shape = item_attr.shape
            attr_dtype = item_attr.dtype.name

            try:
                attr_val = attr_kwargs[attr_name]
            except KeyError:
                raise RuntimeError(f"Item attribute missing: {attr_name}")

            if attr_shape is not None and np.array(attr_val).shape != attr_shape:
                raise ValueError(
                    f"Shape mismatch for {attr_name} "
                    f"(expected: {attr_shape}, "
                    f"got: {np.array(attr_val).shape})"
                )

            attr_val = (
                np.expand_dims(attr_val, axis=0)
                if isinstance(attr_val, np.ndarray)
                else np.array([attr_val], dtype=attr_dtype)
            )
            attr_val = attr_val.astype(attr_dtype)

            item_dict[attr_name] = attr_val

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
    def read_metadata(filepath: FilepathType) -> "MetadataView":
        filepath = Path(filepath).resolve()
        with zarr.ZipStore(filepath, mode="r") as store:
            root = zarr.group(store=store)
            metadata = root.attrs[_METADATA_KEY]
        return MetadataViewMeta(sorted(metadata.keys()))(**metadata)

    @classmethod
    def read_item_attrs(cls, filepath: FilepathType) -> "ItemAttrSetView":
        filepath = Path(filepath).resolve()
        with zarr.ZipStore(filepath, mode="r") as store:
            root = zarr.group(store=store)
            item_attrs_serialized = root.attrs[_ITEMATTRS_KEY]

        item_attrs = list(map(ItemAttr.deserialize, item_attrs_serialized))
        return ItemAttrSetViewMeta(map(lambda a: a.name, item_attrs))(*item_attrs)

    @classmethod
    def build(cls, filepath: FilepathType) -> "DatasetBuilder":
        filepath = Path(filepath).resolve()
        if filepath.exists():
            raise FileExistsError(filepath)

        return DatasetBuilder(filepath)


class DatasetBuilder:
    def __init__(self, filepath: FilepathType):
        self._lock = Lock()
        self._path = filepath
        self._metadata = dict()
        self._item_attrs = list()

    def _setup_dataset(self):
        item_attrs_serialized = list(map(lambda it: it.serialize(), self._item_attrs))

        with self._lock, zarr.ZipStore(self._path, mode="w") as store:
            root = zarr.group(store=store)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)

                root.attrs[_METADATA_KEY] = dict(self._metadata)
                root.attrs[_ITEMATTRS_KEY] = item_attrs_serialized

            for item_attr_serialized in item_attrs_serialized:
                attr_name = item_attr_serialized["name"]
                attr_dtype = item_attr_serialized["dtype_name"]
                attr_shape = (
                    tuple(item_attr_serialized["shape"])
                    if item_attr_serialized["shape"] is not None
                    else None
                )

                collection_shape = (0,) + attr_shape if attr_shape is not None else (0,)
                chunk_shape = (1,) + attr_shape if attr_shape is not None else (1,)

                root.create_dataset(
                    attr_name,
                    shape=collection_shape,
                    chunks=chunk_shape,
                    dtype=attr_dtype,
                )

    def with_metadata(self, **metadata_kwargs) -> "DatasetBuilder":
        self._metadata.update(metadata_kwargs)
        return self

    def add_item_attr(
        self,
        name: str,
        shape: Union[Tuple[int, ...], None],
        dtype: Union[str, np.dtype],
    ) -> "DatasetBuilder":

        self._item_attrs.append(ItemAttr(name=name, shape=shape, dtype=dtype))
        return self

    def conclude_build(self) -> Dataset:
        self._setup_dataset()
        return Dataset(self._path)
