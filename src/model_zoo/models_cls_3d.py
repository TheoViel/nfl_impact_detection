import torch
import torch.nn as nn

from model_zoo.resnet_3d import generate_model, forward_with_aux_resnet_3d
from model_zoo.i3d import InceptionI3d
from model_zoo.configs import CONFIGS
from utils.torch import load_model_weights_3d, load_model_weights
from params import DATA_PATH

import sys
sys.path.append('../mmaction2/')
from mmaction.models import build_model  # noqa
from mmcv.runner import load_checkpoint  # noqa


CP_PATHS = {
    "resnet18": DATA_PATH + "weights/r3d18_KM_200ep.pth",
    "resnet34": DATA_PATH + "weights/r3d34_KM_200ep.pth",
    "resnet50": DATA_PATH + "weights/r3d50_KMS_200ep.pth",
    "i3d": DATA_PATH + "weights/rgb_imagenet.pt",
    "slowfast": DATA_PATH + "weights/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb.pth",
    "slowonly": DATA_PATH + "weights/slowonly_r50_omni_4x16x1_kinetics400_rgb.pth",
}


def get_model_cls_3d(name, num_classes=1, num_classes_aux=0, pretrained=True):
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
    if "resnet" in name:
        n_classes = 1139 if "KMS" in CP_PATHS[name] else 1039
        depth = int(name[-2:])
        model = generate_model(depth, n_classes=n_classes)

        if pretrained:
            load_model_weights_3d(model, CP_PATHS[name])
    elif name == "i3d":  # i3d
        model = InceptionI3d(num_classes=400, in_channels=3)

        if pretrained:
            load_model_weights(model, CP_PATHS[name])

    elif name in ["slowfast", "slowonly"]:
        model = build_model(CONFIGS[name])

        if pretrained:
            print(f'\n -> Loading weighs from "{CP_PATHS[name]}"\n')
            load_checkpoint(model, CP_PATHS[name])
    else:
        raise NotImplementedError

    model.name = name
    model.num_classes = num_classes
    model.num_classes_aux = num_classes_aux

    if "resnet" in name:
        # Strides
        model.conv1.stride = (1, 1, 1)
        model.layer2[0].conv1.stride = (1, 2, 2)
        model.layer2[0].downsample[0].stride = (1, 2, 2)
        model.layer3[0].conv1.stride = (1, 2, 2)
        model.layer3[0].downsample[0].stride = (1, 2, 2)
        model.layer4[0].conv1.stride = (1, 2, 2)
        model.layer4[0].downsample[0].stride = (1, 2, 2)
        model.maxpool.stride = (1, 2, 2)

        model.nb_ft = model.fc.in_features
        model.fc = nn.Linear(model.nb_ft, num_classes)
        model.forward = lambda x: forward_with_aux_resnet_3d(model, x)

        if num_classes_aux:
            model.fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    elif name == "i3d":
        model.Conv3d_1a_7x7.conv3d.stride = (1, 1, 1)
        # model.MaxPool3d_2a_3x3.stride = (1, 1, 1)
        model.MaxPool3d_4a_3x3.stride = (1, 2, 2)
        model.MaxPool3d_5a_2x2.stride = (1, 2, 2)

        model.nb_ft = model.logits.conv3d.in_channels
        model.logits = nn.Linear(model.nb_ft, num_classes)

        if num_classes_aux:
            model.fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    elif name == "slowfast":
        model.backbone.slow_path.conv1.stride = (1, 1, 1)
        model.backbone.fast_path.conv1.stride = (1, 1, 1)

        model.backbone.slow_path.maxpool.stride = (1, 1, 1)
        model.backbone.fast_path.maxpool.stride = (1, 1, 1)

        model.backbone.slow_path.pool2.stride = (1, 1, 1)
        model.backbone.fast_path.pool2.stride = (1, 1, 1)

        model.dropout = nn.Dropout(0.5)
        model.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        model.nb_ft = model.cls_head.fc_cls.in_features

        model.fc = nn.Linear(model.nb_ft, num_classes)
        model.forward = lambda x: forward_slowfast(model, x)

        if num_classes_aux:
            model.fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    else:
        model.backbone.conv1.stride = (1, 1, 1)
        model.backbone.pool2.stride = (1, 1, 1)
        model.backbone.maxpool.stride = (1, 1, 1)

        model.dropout = nn.Dropout(0.5)
        model.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        model.nb_ft = model.cls_head.fc_cls.in_features

        model.fc = nn.Linear(model.nb_ft, num_classes)
        model.forward = lambda x: forward_slowonly(model, x)

        if num_classes_aux:
            model.fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    return model


def forward_slowfast(self, x):
    x = x[:, :, 1:, :, :]  # size BS x 8 x C x W x H
    ft1, ft2 = self.extract_feat(x)

    ft1 = self.avg_pool(ft1).view(x.size(0), -1)
    ft2 = self.avg_pool(ft2).view(x.size(0), -1)

    ft = torch.cat([ft1, ft2], -1)
    ft = self.dropout(ft)

    y = self.fc(ft)

    if self.num_classes_aux > 0:
        y_aux = self.fc_aux(ft)
        return y, y_aux

    return y, 0


def forward_slowonly(self, x):
    ft = self.extract_feat(x)
    # print(ft.size())

    ft = self.avg_pool(ft).view(x.size(0), -1)
    ft = self.dropout(ft)

    y = self.fc(ft)

    if self.num_classes_aux > 0:
        y_aux = self.fc_aux(ft)
        return y, y_aux

    return y, 0
