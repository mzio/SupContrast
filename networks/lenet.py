import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet4(nn.Module):
    """
    Adapted from https://github.com/activatedgeek/LeNet-5
    """
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get('num_channels', 1)
        classes = kwargs.get('num_classes', 10)
        self.convnet = nn.Sequential(
            OrderedDict([('c1', nn.Conv2d(in_channels, 6, kernel_size=(5, 5))),
                         ('relu1', nn.ReLU()), ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                         ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))), ('relu3', nn.ReLU()),
                         ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                         ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))), ('relu5', nn.ReLU())]))

        self.activation_layer_name = 'convnet.relu5'
        self.fc = nn.Linear(120, classes)

    def forward(self, img):
        x = self.convnet(img)
        x = x.view(x.size(0), -1)
        out = x
        return out
    
class SupConLeNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='lenet', head='mlp', feat_dim=128):
        super(SupConLeNet, self).__init__()
        model_fun, dim_in = (LeNet4, 120)
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat