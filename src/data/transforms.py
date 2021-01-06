import cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

from params import MEAN, STD

cv2.setNumThreads(0)


def blur_transforms(p=0.5):
    """
    Applies MotionBlur, GaussianBlur or RandomFog random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(blur_limit=3),
            albu.GaussianBlur(blur_limit=(3, 5)),
            albu.Blur(blur_limit=3),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(70, 130)),
            albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            albu.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        ],
        p=p,
    )


def channel_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.ChannelShuffle(p=0.9),
            albu.ChannelDropout(p=0.1),
            # albu.ToGray(p=0.1),
        ],
        p=p,
    )


def size_transforms(p=0.5, size=512):
    """
    Applies RandomSizedCrop or Resize or RandomCrop with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomSizedCrop(min_max_height=(512, 720), height=size, width=size, p=0.5),
            albu.Resize(size, size),
            albu.RandomCrop(size, size),
        ],
        p=p,
    )


def get_transfos_det(visualize=False, train=True):

    bbox_params = albu.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
    )

    if visualize:
        normalizer = albu.Compose(
            [
                ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [
                albu.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ],
            p=1,
        )

    if train:
        return albu.Compose(
            [
                size_transforms(p=1),
                albu.Rotate(limit=45, p=0.5),
                color_transforms(p=0.5),
                channel_transforms(p=0.1),
                # blur_transforms(p=0.5),
                albu.HorizontalFlip(p=0.5),
                normalizer,
            ],
            bbox_params=bbox_params,
        )

    else:
        return albu.Compose(
            [
                albu.Resize(512, 512),
                normalizer,
            ],
            bbox_params=bbox_params,
        )


def get_transfos_cls(visualize=False, train=True):
    if visualize:
        normalizer = albu.Compose(
            [
                ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [
                albu.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ],
            p=1,
        )

    if train:
        return albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.5, shift_limit=0.5, rotate_limit=90, p=0.75
                ),
                color_transforms(p=0.5),
                channel_transforms(p=0.2),
                normalizer,
            ],
        )

    else:
        return normalizer


def get_transfos_cls_3d(visualize=False, train=True):
    if visualize:
        normalizer = albu.Compose(
            [
                ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [
                albu.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ],
            p=1,
        )

    if train:
        return albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.5, shift_limit=0.5, rotate_limit=90, p=0.75
                ),
                color_transforms(p=0.5),
                channel_transforms(p=0.2),
                normalizer,
            ],
        )

    else:
        return normalizer
