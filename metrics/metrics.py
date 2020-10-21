#-*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F
import torch

#=============================================================================
#
#     metrics.py
#
#     定义评价指标函数，其中compute_IoU用于计算两个bbox的IoU，compute_class_acc计算
#     图像分类正确率，compute_iou_acc计算图像定位正确率和同时分类定位正确率
#
#=============================================================================


def compute_IoU(boxes1, boxes2):
    """
    功能：
      计算IoU
    参数：
      boxes1: 真实的bbox (n,4)
      boxes2: 预测的bbox (n,4)
    返回：
      iou：IoU计算结果 （n,）
    """
    xA = torch.max(boxes1[:, 0], boxes2[:, 0])  # (N,)
    yA = torch.max(boxes1[:, 1], boxes2[:, 1])  # (N,)
    xB = torch.min(boxes1[:, 2], boxes2[:, 2])  # (N,)
    yB = torch.min(boxes1[:, 3], boxes2[:, 3])  # (N,)
    interArea = (xB - xA + 1) * (yB - yA + 1)
    # 无交集，设为0
    interArea[xA > xB] = 0

    boxAArea = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    boxBArea = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def compute_class_acc(in_class, gt_class):
    """
    功能：
      计算分类正确的个数，在该函数中不计算正确率，
      在训练或测试完成后统一除样本数完成正确率的计算
    参数：
      in_class: 预测的类别 (n,)
      gt_class: 真实的类别 (n,)
    返回：
      class_acc：正确分类的个数
    """
    _, pred_class = in_class.max(dim =1)  # pred_class :(n,),取值0~4
    class_acc = pred_class.eq(gt_class).sum().item()
    return class_acc


def compute_iou_acc(in_class, gt_class, in_coor, gt_coor, theta=0.5):
    """
    功能：
      计算平均IoU，图像定位正确率和同时分类定位正确率
      设置IoU阈值为0.5  <0.5不将其算入定位准确率
    参数：
      in_class: 预测的类别 (n,)
      gt_class: 真实的类别 (n,)
      in_coor: 预测的bbox (n,4)
      gt_coor: 真实的bbox (n,4)
      theta: IoU阈值
    返回：
      class_acc：同时分类定位正确率
      mean_IoU：平均IoU
      Loc_Acc：定位准确率
    """
    in_coor[in_coor < 0] = 0
    in_coor[in_coor > 1] = 1
    IoU = compute_IoU(gt_coor*128, in_coor*128)  # (n,)
    mean_IoU = IoU.sum().item()
    IoU_count = IoU
    IoU_count[IoU < 0.5] = 0
    IoU_count[IoU >= 0.5] = 1
    Loc_Acc = IoU.sum().item()
    if len(in_class.size()) == 1:
        class_acc = in_class.eq(gt_class).sum().item()
    else:
        _, pred_class = in_class.max(dim=1)  # pred_class :(n,),取值0~4
        # TODO: <0.5不将其算入分类准确率
        pred_class[IoU < 0.5] = 100
        class_acc = pred_class.eq(gt_class).sum().item()
    return class_acc, mean_IoU, Loc_Acc
