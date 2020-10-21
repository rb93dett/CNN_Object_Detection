#-*-coding:utf-8-*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw

#=============================================================================
#
#     display.py
#
#     实现目标识别的可视化展示，包括图像，目标bbox，预测的bbox和目标类别名称
#
#=============================================================================


def dis_gt(img, name, coor):
    """
    功能：
      实现目标识别的可视化展示，包括图像，目标bbox，预测的bbox和目标类别名称
    参数：
        img： 输入图像
        name: 目标类别或者list, 对应坐标列表
        coor: np.array一维数组，四个坐标，依次为（xmin, ymin, xmax, ymax）
              或者一个列表，[ [x1min, y1min, x1max, y1max],[x2min, y2min, x2max, y2max] ]
              其中第一个为预测坐标，第二个为实际坐标
    返回：
        显示图片
    """
    draw = ImageDraw.Draw(img)
    color = ["blue", "magenta"]
    if isinstance(coor, list):
        for loc in range(len(coor)):  # (0,1)
            draw.text((coor[loc][0], coor[loc][1]), name[loc], (255, 255, 255))
            draw.rectangle((coor[loc][0], coor[loc][1], coor[loc][2], coor[loc][3]), outline=color[loc])
    else:
        draw.text((coor[0], coor[1]), name, (255, 255, 255))
        draw.rectangle((coor[0], coor[1], coor[2], coor[3]), outline=color[0])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # test
    dis_gt(Image.open("./tiny_vid/bird/000001.JPEG"), 'bird', np.array([46, 0, 123, 70]))
