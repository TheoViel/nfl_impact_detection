import sys
from params import SIZE

sys.path.append("../timm-efficientdet-pytorch")
from effdet import ( # noqa
    get_efficientdet_config,
    EfficientDet,
    DetBenchTrain,
    DetBenchEval,
)
from effdet.efficientdet import HeadNet  # noqa
from effdet.helpers import load_pretrained  # noqa


def get_model(name, num_classes=2):
    config = get_efficientdet_config(name)
    net = EfficientDet(config, pretrained_backbone=False)

    load_pretrained(net, config.url)

    config.num_classes = num_classes
    config.image_size = SIZE  # // 2

    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
    )

    model = DetBenchTrain(net, config)
    model.name = name
    model.num_classes = num_classes

    return model


def get_val_model(train_model):
    config = get_efficientdet_config(train_model.name)
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = train_model.num_classes
    config.image_size = SIZE

    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
    )

    state_dict = train_model.model.state_dict()
    net.load_state_dict(state_dict)

    return DetBenchEval(net, config).eval()


def get_inference_model(name, num_classes=2):
    config = get_efficientdet_config(name)
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = num_classes
    config.image_size = SIZE

    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
    )

    return DetBenchEval(net, config).eval()
