# -*-coding:utf-8-*-

from __future__ import print_function
import numpy as np

import torch
from PIL import Image
from model._init_ import Net
from manage_data._init_ import xywh_to_x1y1x2y2, x1y1x2y2_to_xywh
from torchvision import transforms
from display import dis_gt

#=============================================================================
#
#     eval.py
#
#     对单张图像进行识别，并可视化输出预测结果，用于测试模型效果
#
#=============================================================================

resume_path = 'trained_model/best_model.pkl'
print('Loading model...')
net = Net('mobilenet', freeze_basenet=False)
net.load_state_dict(torch.load(resume_path, map_location='cpu')["model_state"])
net.eval()

print('load image...')
img_path = './test_img/000036.JPEG'
target_classes = ['car', 'bird', 'turtle', 'dog', 'lizard']
bbox = np.array([39, 51, 117, 89], dtype=np.float32)
label = np.array([0], dtype=np.int32)

img = Image.open(img_path).convert('RGB')
IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])(img)
# gt_class,gt_bbox =torch.Tensor(label),torch.Tensor(bbox/128.)

with torch.no_grad():
    outputs_reg, outputs_class = net(IMG.unsqueeze(0))
    print(outputs_reg,outputs_class)
    _, pred_label = outputs_class.squeeze(0).max(dim=0)
    pred_bbox = xywh_to_x1y1x2y2(outputs_reg).squeeze(0)
    print(pred_label, pred_bbox)

dis_gt(img, [target_classes[int(pred_label.item())], target_classes[label[0]]], [pred_bbox.numpy()*128, bbox])
