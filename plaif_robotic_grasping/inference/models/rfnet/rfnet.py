import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grasp_model import GraspModel

from itertools import chain # 串联多个迭代对象
from .util import _BNReluConv, upsample
from .resnet.resnet_single_scale_single_attention import *

class GenerativeResnet(GraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=True, prob=0.0):
        super(GenerativeResnet, self).__init__()
        use_bn = True
        # self.  backbone = resnet18(pretrained=True, efficient=False, use_bn= True)
        self.  backbone = resnet34(pretrained=False, efficient=False, use_bn= True)
        self.logits = _BNReluConv(self.backbone.num_features, 4, batch_norm=use_bn)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)
        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)
        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

    def forward(self, x_in):
        x_dep = x_in[:,0, ...]
        x_rgb = x_in[:,1:, ...]
        
        ## rfnet
        x, additional = self.backbone(x_rgb, x_dep)

        ## gr-conv deconv
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output


    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()