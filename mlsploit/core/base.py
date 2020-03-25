from pydantic import BaseModel


__all__ = ["FauxImmutableModel"]


class FauxImmutableModel(BaseModel):
    # pylint: disable=too-few-public-methods
    class Config:
        allow_mutation = False
