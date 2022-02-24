import torch
from torch import nn
import torch.nn.functional as F
from model.utils import Hourglass, make_coordinate_grid


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion=32, num_kp=4, num_channels=3, max_features=2048,
                 num_blocks=6, temperature=0.1, scale_factor=1, pad=0):
        super(KPDetector, self).__init__()
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)
        self.temperature = temperature
        self.scale_factor = scale_factor

    def gaussian2kp(self, heatmap):
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(
            shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        # kp = {'value': value}

        return value

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)
        out = self.gaussian2kp(heatmap)
        return out
