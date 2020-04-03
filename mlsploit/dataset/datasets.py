from .base import Dataset, Feature
from .features import ClassLabelFeature, ImageDataFeature


__all__ = [
    "ImageDataset",
    "ImageClassificationDataset",
    "ImageClassificationDatasetWithPrediction",
]


class ImageDataset(Dataset):
    filename = Feature(shape=None, dtype=str)
    image = ImageDataFeature(
        image_size=224, color_mode="RGB", channels_first=True, bounds=(0, 1)
    )

    class DefaultMetadata:
        # pylint: disable=too-few-public-methods
        module = ""


class ImageClassificationDataset(ImageDataset):
    label = ClassLabelFeature(num_classes=1000)


class ImageClassificationDatasetWithPrediction(ImageClassificationDataset):
    prediction = ClassLabelFeature(num_classes=1000)
