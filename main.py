import torchvision.transforms
from utils.weight_grid import show_conv_weight_grid
from model.make_model import make_vgg19, make_alexnet
import torch
from torchvision import utils
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pretrained_vgg19 = make_vgg19()
    pretrained_alexnet = make_alexnet()

    """
    if visualize_type == 'avg', visualize all filters of any layer with avg.
    elif visualize_type == 'one', visualize one filters of any layer.
    """
    show_conv_weight_grid(pretrained_alexnet.features, 1, visualize_type='avg', nrow='auto',
                            save=True, save_name='alexnet_conv1_avg')

    show_conv_weight_grid(pretrained_alexnet.features, 1, visualize_type='one', filter_index=10, nrow=3,
                            save=True, save_name='alexnet_conv1_10')

    show_conv_weight_grid(pretrained_alexnet.features, 3, visualize_type='one', filter_index=5, nrow='auto',
                            save=True, save_name='alexnet_conv3_5')

    show_conv_weight_grid(pretrained_vgg19.features, 12, visualize_type='avg', nrow='auto',
                          save=True, save_name='vgg19_conv12_avg')

    show_conv_weight_grid(pretrained_vgg19.features, 7, visualize_type='one', filter_index=56, nrow='input',
                          save=True, save_name='vgg19_conv7_56')