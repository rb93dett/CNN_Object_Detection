import torch.nn as nn
import torch.nn.functional as F

#=============================================================================
#
#     mobilenet.py
#
#     构造MobileNet网络架构
#
#=============================================================================

class MobileNet(nn.Module):
    # input 3*128*128 output 1024*4*4
    def __init__(self, num_classes=1024):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.base_net = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),       # 11
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),     # 13
        )

    def forward(self, x):
        x = self.base_net(x)
        # x = F.avg_pool2d(x, 7)
        # x = x.view(-1, 1024)
        # x = self.fc(x)
        return x
