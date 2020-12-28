import torch
import torchvision
import numpy as np
import torch.nn as nn
import resnest.torch as resnest_torch

from params import NUM_WORKERS
from model_zoo.models_cls import RESNETS, forward_with_aux_efficientnet, forward_with_aux_resnet
from utils.torch import load_model_weights
from data.transforms import get_transfos_cls
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

from inference.classifier import NFLDatasetClsInference


def get_model_cls_aux(
    name,
    num_classes=1,
    num_classes_aux=0,
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
    model.num_classes = num_classes
    model.num_classes_aux = num_classes_aux

    if "efficientnet" not in name:
        model.conv1.stride = (1, 1)

        model.nb_ft = model.fc.in_features
        model.fc = nn.Linear(model.nb_ft, num_classes)
        model.forward = lambda x: forward_with_aux_resnet(model, x)

        if num_classes_aux:
            model.fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    else:
        model._conv_stem.stride = (1, 1)

        model.nb_ft = model._fc.in_features
        model._fc = nn.Linear(model.nb_ft, num_classes)

        model.forward = lambda x: forward_with_aux_efficientnet(model, x)

        if num_classes_aux:
            model._fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    return model


def retrieve_model_aux(config, fold=0, log_folder=""):
    model = get_model_cls_aux(
        config["name"],
        num_classes=config["num_classes"],
        num_classes_aux=config["num_classes_aux"],
    ).eval()
    model.zero_grad()

    model = load_model_weights(model, log_folder + f"{config['name']}_{fold}.pt")

    return model


def inference_aux(df, models, batch_size=256, device="cuda", root=""):
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
    preds_aux = []
    with torch.no_grad():
        for img in loader:
            img = img.to(device)
            preds_img = []
            preds_aux_img = []

            for model in models:
                y_pred, y_pred_aux = model(img)

                preds_img.append(y_pred.sigmoid().detach().cpu().numpy())
                preds_aux_img.append(y_pred_aux.softmax(-1).detach().cpu().numpy())

            preds.append(np.mean(preds_img, 0))
            preds_aux.append(np.mean(preds_aux_img, 0))

    return np.concatenate(preds), np.concatenate(preds_aux)
