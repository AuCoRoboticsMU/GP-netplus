from builtins import super

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FcnResnet50(nn.Module):
    def __init__(self):
        """
        Initiate a FCN ResNet-50 model from pytorch pre-trained on Imagenet V2.
        """
        super(FcnResnet50, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.segmentation.fcn_resnet50(weights_backbone=weights, num_classes=6)

    def forward(self, x):
        """
        Run a forward pass of the FCN ResNet-50 model.
        :param x: Model input. In the case of GP-Net+, this should be an RGB image with 3 channels (HxWx3).
        :return: Predicted quality (HxWx1), Orientation (HxWx4) and Width (HxWx1).
        """

        y = self.model(x)['out']

        qual_out = torch.sigmoid(y[:, 0:1, :, :])
        rot_out = F.normalize(y[:, 1:5, :, :], dim=1)
        width_out = torch.sigmoid(y[:, 5:, :, :])
        return qual_out, rot_out, width_out


def load_network(path, device):
    """
    Construct the neural network and load parameters from the specified file.

    :param path: Path to .pt file to load weights from.
    :param device: Device to use for the tensors. Can be GPU or CPU.
    :return: a FCN ResNet-50 model with weights used for GP-Net+.
    """
    net = FcnResnet50().to(device)

    try:
        print(type)
        net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(path, map_location=device))
        net = net.module
    net.eval()
    return net
