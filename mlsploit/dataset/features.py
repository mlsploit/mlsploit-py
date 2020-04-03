import json
from typing import Tuple, Union

from typing_extensions import Literal

from .base import Feature, Metadata


__all__ = ["ClassLabelFeature", "ImageDataFeature"]


class ClassLabelFeature(Feature):
    def __init__(self, num_classes: int):
        """Feature for storing class label

        Args:
            num_classes (int):
                Total number of classes
        """
        super().__init__(
            shape=None, dtype=int, metadata=Metadata(num_classes=num_classes)
        )

    @classmethod
    def deserialize(cls, data: str) -> "ClassLabelFeature":
        m = json.loads(data)["metadata"]
        return cls(num_classes=m["num_classes"])


class ImageDataFeature(Feature):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        color_mode: Literal["RGB", "BGR", "GRAY"] = "RGB",
        channels_first: bool = False,
        bounds: Tuple[int, int] = (0, 255),
    ):
        """Feature for storing image data

        Args:
            image_size (int or Tuple[int, int]):
                Size of an image as int or (width, height)
            color_mode (str, optional):
                RGB or BGR or GRAY, default = RGB
            channels_first (bool, optional):
                Whether images are stored in channels-first mode, default = False
            bounds (Tuple[int, int], optional):
                Tuple of minimum and maximum values (not enforced), default = (0, 255)
        """

        num_channels = 1 if color_mode == "GRAY" else 3
        width, height = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )

        shape = (
            (num_channels, height, width)
            if channels_first
            else (height, width, num_channels)
        )

        super().__init__(
            shape=shape,
            dtype=float,
            metadata=Metadata(
                image_width=width,
                image_height=height,
                color_mode=color_mode,
                channels_first=channels_first,
                bounds=json.dumps(bounds),
            ),
        )

    @classmethod
    def deserialize(cls, data: str) -> "ImageDataFeature":
        m = json.loads(data)["metadata"]
        return cls(
            image_size=(m["image_width"], m["image_height"]),
            color_mode=m["color_mode"],
            channels_first=m["channels_first"],
            bounds=json.loads(m["bounds"]),
        )
