import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
import resnest.torch as resnest_torch

from params import NUM_WORKERS
from model_zoo.models_cls import RESNETS
from utils.torch import load_model_weights
from data.transforms import get_transfos_cls
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset


def extend_box(box, size=64):
    w = box[1] - box[0]
    h = box[3] - box[2]

    dw = (size - w) / 2
    dh = (size - h) / 2

    new_box = [
        box[0] - np.floor(dw),
        box[1] + np.ceil(dw),
        box[2] - np.floor(dh),
        box[3] + np.ceil(dh),
    ]
    return np.array(new_box).astype(int)


def adapt_to_shape(box, shape):
    if box[0] < 0:
        box[1] -= box[0]
        box[0] = 0
    elif box[1] >= shape[1]:
        diff = box[1] - shape[1]
        box[1] -= diff
        box[0] -= diff

    if box[2] < 0:
        box[3] -= box[2]
        box[2] = 0

    elif box[3] >= shape[0]:
        diff = box[3] - shape[0]
        box[3] -= diff
        box[2] -= diff

    return box


class NFLDatasetClsInference(Dataset):
    def __init__(self, df, transforms=None, root=""):
        super().__init__()
        self.transforms = transforms
        self.root = root

        self.images = np.unique(df["image_name"].values)
        self.images = [cv2.imread(self.root + img) for img in self.images]

        self.frame_to_img = list(np.unique(df["frame"].values))
        self.frames = df["frame"].values

        self.boxes = df[["left", "width", "top", "height"]].values
        self.boxes[:, 1] += self.boxes[:, 0]
        self.boxes[:, 3] += self.boxes[:, 2]

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        frame = self.frame_to_img.index(self.frames[idx])
        image = self.images[frame]

        box = extend_box(self.boxes[idx], size=64)
        box = adapt_to_shape(box, image.shape)

        image = image[box[2]: box[3], box[0]: box[1]]

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image


def get_model_cls(
    name,
    num_classes=1,
):
    """
    Loads a pretrained model.
    Supports Resnet based models.

    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.

    Raises:
        NotImplementedError: Specified model name is not supported.

    Returns:
        torch model -- Pretrained model.
    """

    # Load pretrained model
    if "resnest" in name:
        model = getattr(resnest_torch, name)(pretrained=False)
    elif name in RESNETS:
        model = getattr(torchvision.models, name)(pretrained=False)
    elif "efficientnet" in name:
        model = EfficientNet.from_name(name)
    else:
        raise NotImplementedError
    model.name = name

    if "efficientnet" not in name:
        model.conv1.stride = (1, 1)

        model.nb_ft = model.fc.in_features
        model.fc = nn.Linear(model.nb_ft, num_classes)

    else:
        model._conv_stem.stride = (1, 1)

        model.nb_ft = model._fc.in_features
        model._fc = nn.Linear(model.nb_ft, num_classes)

    model.num_classes = num_classes

    return model


def retrieve_model(config, fold=0, log_folder=""):
    model = get_model_cls(
        config["name"],
        num_classes=config["num_classes"],
    ).eval()
    model.zero_grad()

    model = load_model_weights(model, log_folder + f"{config['name']}_{fold}.pt")

    return model


def inference(df, models, batch_size=256, device="cuda", root=""):
    models = [model.to(device).eval() for model in models]

    dataset = dataset = NFLDatasetClsInference(
        df.copy(),
        transforms=get_transfos_cls(train=False),
        root=root,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds = []
    with torch.no_grad():
        for img in loader:
            img = img.to(device)
            preds_img = []

            for model in models:
                y_pred = model(img)
                preds_img.append(y_pred.sigmoid().detach().cpu().numpy())

            preds.append(np.mean(preds_img, 0))

    return np.concatenate(preds)
