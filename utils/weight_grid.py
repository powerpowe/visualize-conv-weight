import torchvision.transforms
from torchsummary import summary
from model.make_model import make_vgg19
import torch
import torch.nn as nn
from torchvision import utils
import matplotlib.pyplot as plt
from PIL import Image

def show_conv_weight_grid(model_layers, layer_num, visualize_type='avg', filter_index=None, nrow='auto',
                            show=True, save=False, save_name=None):  # feature layers
    """make weight grid of pre-trained VGG19 (no bn)

        Args:
            :model_layers (nn.Sequential): layers of the model that want to make the weight grid

            :layer_num (int): layer to check

            :visualize_type ('one', 'avg'): determine how to visualize

                if visualize_type == one, visualize only one filter of layer (#filter_num)
                elif visualize_type == avg, visualize all filters of layer with avg

            :filter_index (None, int): determine which filter to visualize

                if visualize_type == avg, It is not used.

            :nrow('auto', 'input', int): how to determine nrow

                if nrow == 'auto', determine nrow by conv_nrow_dict,
                                   It only works when the {channel, filter_num}is in [16, 32, 64, 128, 256, 512, 1024]
                                   ({channel, filter_num} is determined by type)
                                   if not in [16, 32, 64, 128, 256, 512, 1024], change nrow to 'input'

                elif nrow == 'input', determine nrow by input

                elif nrow is int, directly determine nrow

            :show (bool): determine whether to show the results

            :save (bool): determine whether to save the results

            :save_name (None, str): determines the name to save

                if save == False, It is not used.

        Returns:
            None
    """
    def _normalize(weight):
        weight_max = torch.max(weight)
        weight_min = torch.min(weight)
        temp = (weight + weight_min) / (weight_max + weight_min)
        return temp

    def _make_grid(weight, nrow=nrow):
        grid_size = weight.shape[0]
        temp = [weight[i] for i in range(grid_size)]
        change = False
        if nrow == 'auto':
            if grid_size in conv_nrow_dict.keys():
                temp_grid = utils.make_grid(temp, nrow=conv_nrow_dict[grid_size], padding=1)
            else:
                change = True

        if nrow == 'input' or change:
            row = int(input(f'num of filters (or channels) is {grid_size}, please enter int for nrow.'))
            temp_grid = utils.make_grid(temp, nrow=row, padding=1)

        if isinstance(nrow, int):
            temp_grid = utils.make_grid(temp, nrow=nrow, padding=1)

        img = torchvision.transforms.ToPILImage()(temp_grid)
        if save:
            img.save(f'./weight_jpg/{save_name}.png')
        if show:
            img.show()

    conv_nrow_dict = {16: 4, 32:4, 64: 8, 128: 8, 256: 16, 512: 16, 1024: 32}

    cnt = 0
    for layer in model_layers:
        if isinstance(layer, nn.Conv2d):
            cnt += 1
            if cnt == layer_num:
                p = layer.parameters()
                weight = next(p)
                num_filter = weight.shape[0]
                num_channel = weight.shape[1]

                if visualize_type == 'avg':
                    temp = torch.mean(weight, dim=1)
                    temp = _normalize(temp)
                    temp = temp.unsqueeze(1)
                    _make_grid(temp, nrow)

                elif visualize_type == 'one':
                    temp = weight[filter_index]
                    temp = _normalize(temp)
                    temp = temp.unsqueeze(1)
                    _make_grid(temp, nrow)

