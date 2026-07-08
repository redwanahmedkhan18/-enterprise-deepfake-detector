"""
Augmentation pipelines. Built with albumentations, kept separate from the
Dataset classes so training code can swap augmentation strategy without
touching data-loading logic.

Torch/albumentations are imported lazily (function-local) so importing this
module — or anything that imports it transitively — doesn't hard-require
those (heavy) packages just to, say, build a manifest.
"""
from ai_engine.config import AugmentationConfig


def build_train_transform(aug_config: AugmentationConfig, image_size: int = 224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.HorizontalFlip(p=aug_config.horizontal_flip_p),
            A.ImageCompression(
                quality_range=aug_config.jpeg_quality_range, p=aug_config.jpeg_compression_p
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=aug_config.gaussian_blur_p),
            A.RandomBrightnessContrast(p=aug_config.brightness_contrast_p),
            A.Downscale(scale_range=aug_config.downscale_range, p=aug_config.downscale_p),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: int = 224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
