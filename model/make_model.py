import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import alexnet, AlexNet_Weights

def make_vgg19():
    weights = VGG19_Weights.IMAGENET1K_V1
    model = vgg19(weights=weights)
    model.eval()
    return model


def make_alexnet():
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights)
    model.eval()
    return model

if __name__ == "__main__":
    model = make_vgg19()
    for i, layer in enumerate(model.features):
        for params in layer.parameters():
            print(i, params.shape)