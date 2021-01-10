from .vgg import *
from .resnet import *
from .seresnet import *
from .mobilenet import *


def get_backbone(name, stride=32):
    name = name.lower()
    if name == "vgg11":
        backbone = VGG11()
    elif name == "vgg13":
        backbone = VGG13()
    elif name == "vgg16":
        backbone = VGG16()
    elif name == "vgg19":
        backbone = VGG19()
    elif name == "vgg19_first10":
        backbone = VGG19()
        backbone.features = backbone.features[:-13]
        return backbone
    elif name == "resnet18":
        backbone = ResNet18()
    elif name == "resnet34":
        backbone = ResNet34()
    elif name == "resnet50":
        backbone = ResNet50()
    elif name == "resnet101":
        backbone = ResNet101()
    elif name == "resnet152":
        backbone = ResNet152()
    elif name == "seresnet50":
        backbone = SEResNet50()
    elif name == "seresnet101":
        backbone = SEResNet101()
    elif name == "seresnet152":
        backbone = SEResNet152()
    elif name == "mobilenet_v1":
        backbone = MobileNetV1()
    elif name == "mobilenet_v2":
        backbone = MobileNetV2()
    else:
        raise NotImplementedError(f"'{name}' has not been implemented yet")

    backbone.change_stride(stride)
    return backbone
