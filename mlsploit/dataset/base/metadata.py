from collections import OrderedDict
import json
from typing import AbstractSet, Generator, Mapping, Tuple, Union, ValuesView


__all__ = ["Metadata"]


ALLOWED_PRIMITIVES = (str, int, float, bool)
PrimitiveType = Union[str, int, float, bool]


class Metadata(Mapping[str, PrimitiveType]):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if (not isinstance(k, str)) or (not k.isidentifier()):
                raise ValueError(k)

            if not any(isinstance(v, t) for t in ALLOWED_PRIMITIVES):
                raise ValueError(v)

        self._data = tuple(kwargs.items())

    def __repr__(self) -> str:
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(["%s=%s" % it for it in self._data]),
        )

    def __eq__(self, o):
        return self.serialize() == o.serialize()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Generator[str, None, None]:
        for k, _ in self._data:
            yield k

    def __getitem__(self, k: str) -> PrimitiveType:
        return dict(self._data)[k]

    def __contains__(self, k: object) -> bool:
        return k in self.dict()

    def dict(self) -> OrderedDict:
        return OrderedDict(self._data)

    def keys(self) -> AbstractSet[str]:
        return self.dict().keys()

    def values(self) -> ValuesView[PrimitiveType]:
        return self.dict().values()

    def items(self) -> AbstractSet[Tuple[str, PrimitiveType]]:
        return self.dict().items()

    def serialize(self) -> str:
        return json.dumps(list(self.items()))

    @classmethod
    def deserialize(cls, data: str) -> "Metadata":
        return cls(**OrderedDict(json.loads(data)))
