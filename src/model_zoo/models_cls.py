import torch
import torch.nn as nn
import resnest.torch as resnest_torch
from efficientnet_pytorch import EfficientNet

RESNETS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def get_model_cls(name, num_classes=1, num_classes_aux=0):
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
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name in RESNETS:
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "efficientnet" in name:
        model = EfficientNet.from_pretrained(name)
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


def forward_with_aux_resnet(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    fts = self.avgpool(x)
    fts = torch.flatten(fts, 1)

    y = self.fc(fts)

    if self.num_classes_aux > 0:
        y_aux = self.fc_aux(fts)
        return y, y_aux

    return y, 0


def forward_with_aux_efficientnet(self, inputs):
    x = self.extract_features(inputs)

    x = self._avg_pooling(x)
    x = x.flatten(start_dim=1)
    x = self._dropout(x)

    y = self._fc(x)

    if self.num_classes_aux > 0:
        y_aux = self._fc_aux(x)
        return y, y_aux

    return y, 0
