# -*-coding:utf-8-*-

from model.mobilenet import *
from model.VGG import *
import torch
import torch.nn as nn
import torch.nn.functional as F

#=============================================================================
#
#     model：_init_.py
#
#     这次项目可以使用两种不同的backbone，VGG和MobileNet，get_mobilenet_model和
#     get_vgg_model函数用于构造这两种模型，并根据我们的目标对原始模型进行改造，同时如果
#     存在预训练的模型，则使用预训练模型的网络参数，Net类用于在backbone之后加入自定义的
#     网络结构，实现完整的预测模型，完成图像分类和物体定位
#
#=============================================================================


def get_mobilenet_model(pretrain=True, num_classes=5, requires_grad=False):
    """
    功能：
      返回去掉全连接层的mobilenet
    参数：
      pretrain: 是否存在预训练模型
      num_classes: 分类类别的数量
      requires_grad：是否需要保存梯度信息，不需要训练backbone网络时，设置为False，即不训练这些网络层
    返回：
      model：mobilenet网络
    """
    model = MobileNet()
    # 不训练backbone网络
    for param in model.parameters():
        param.requires_grad = requires_grad

    if pretrain:
        # 加载预训练模型
        basenet_state = torch.load("./pretrained_model/mobilenet.pth", map_location='cpu')
        # 暂存原始网络的参数
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in basenet_state.items() if k in model_dict}
        # 导入预训练模型的网络参数
        model.load_state_dict(pretrained_dict)
        return model
    else:
        return model


def get_vgg_model(vggname='VGG16', pretrain=True, num_classes=5, requires_grad=False):
    """
    功能：
      返回去掉全连接层的VGG16网络
    参数：
      pretrain: 是否存在预训练模型
      num_classes: 分类类别的数量
      requires_grad：是否需要保存梯度信息，不需要训练backbone网络时，设置为False，即不训练这些网络层
    返回：
      model：VGG16网络
    """
    model = VGG(vggname)  # (n,512,4,4)
    # 不训练backbone网络
    for param in model.parameters():
        param.requires_grad = requires_grad

    if pretrain:
        # 加载预训练模型
        basenet_state = torch.load("./pretrained_model/vgg16.pth")
        # 暂存原始网络的参数
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in basenet_state.items() if k in model_dict}
        # 导入预训练模型的网络参数
        model.load_state_dict(pretrained_dict)
        return model
    else:
        return model


class Net(nn.Module):
    """
    功能：
      在backbone之上增加extra layer以完成目标分类和定位任务，实现完整的预测模型
    """
    def __init__(self, netname, freeze_basenet=False):
        super(Net,self).__init__()
        if netname[:3] == 'VGG16':
            self.base_net = get_vgg_model(netname, requires_grad=not freeze_basenet)
            self.in_features = 512
        else:
            self.base_net = get_mobilenet_model(requires_grad=not freeze_basenet)
            self.in_features = 1024

        # 预测四个坐标
        self.model_reg = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 4)
        )
        # 预测分类
        self.model_class = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512,32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 5)
        )

    def forward(self, x):
        out = self.base_net(x)                  # out: (n,in_features,4,4)
        out = F.avg_pool2d(out, 4) 				# out: (n,in_features,1,1)
        out = out.view(-1, self.in_features) 	# out: (n,in_features)
        out_reg = self.model_reg(out) 				# out: (n,4)
        out_class = self.model_class(out)			# out: (n,5)

        return out_reg, out_class
