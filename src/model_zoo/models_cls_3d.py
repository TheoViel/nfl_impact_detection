import torch.nn as nn

from model_zoo.resnet_3d import generate_model
from utils.torch import load_model_weights_3d
from params import DATA_PATH


CP_PATHS = {
    "resnet18": DATA_PATH + "weights/r3d18_KM_200ep.pth"
}


def get_model_cls_3d(name, num_classes=1, num_classes_aux=0):
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
        depth = int(name[-2:])
        model = generate_model(depth, n_classes=1039)
    else:
        raise NotImplementedError

    load_model_weights_3d(model, CP_PATHS[name])

    model.name = name
    model.num_classes = num_classes
    model.num_classes_aux = num_classes_aux

    if "resnet" in name:
        model.conv1.stride = (1, 1, 1)   # TODO : Check stride

        model.nb_ft = model.fc.in_features
        model.fc = nn.Linear(model.nb_ft, num_classes)
        model.forward = lambda x: forward_with_aux_resnet_3d(model, x)

        if num_classes_aux:
            model.fc_aux = nn.Linear(model.nb_ft, num_classes_aux)

    else:
        raise NotImplementedError

    return model


def forward_with_aux_resnet_3d(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    if not self.no_max_pool:
        x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)

    x = x.view(x.size(0), -1)
    y = self.fc(x)

    if self.num_classes_aux > 0:
        y_aux = self.fc_aux(x)
        return y, y_aux

    return y, 0
