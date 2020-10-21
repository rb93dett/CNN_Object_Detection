# -*-coding:utf-8-*-
'''Encode target locations and labels.'''
from __future__ import print_function
import torch

import math
import itertools
import numpy as np


class DataEncoder:
    def __init__(self):
        scale = 300.
        steps = [s / scale for s in (8, 16, 32, 64, 100, 300)]
        sizes = [s / scale for s in (30, 60, 111, 162, 213, 264, 315)]
        aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
        feature_map_sizes = (38, 19, 10, 5, 3, 1)

        num_layers = len(feature_map_sizes)

        boxes = []
        for i in range(num_layers):
            fmsize = feature_map_sizes[i]
            for h,w in itertools.product(range(fmsize), repeat=2):
                cx = (w + 0.5)*steps[i]
                cy = (h + 0.5)*steps[i]

                s = sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(sizes[i] * sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = sizes[i]
                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        self.default_boxes = torch.Tensor(boxes)

    def iou(self, box1, box2):
        '''
        功能：
          计算IoU
        参数:
          box1: 物体边框bbox, 维度 [N,4].
          box2: 物体边框bbox, 维度 [N,4].
        返回:
          iou计算结果
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
