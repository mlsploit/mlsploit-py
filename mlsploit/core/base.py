from pathlib import Path
from typing import Type, Union

from pydantic import BaseModel


__all__ = ['FauxImmutableModel']


class FauxImmutableModel(BaseModel):
    class Config:
        allow_mutation = False
