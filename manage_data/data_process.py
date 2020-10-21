from __future__ import print_function
import torch,os,sys,random,cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from matplotlib import pyplot as plt
from encoder import DataEncoder
from torch.utils import data
from torchvision import transforms
from os.path import join as pjoin

import sys
sys.path.insert(0, '../../')

#=============================================================================
#
#     data_process.py
#
#     对图像和标签数据进行预处理，其中函数train_test_txt用于生成我们指定格式的标签数据，
#     tiny_vid_loader用于对原始图像进行augmentation，包括图像变形，翻转和裁切，从而
#     增加数据量，提高模型效果
#
#=============================================================================


def train_test_txt(defualt_path='../tiny_vid'):
    """
    功能：
      以如下格式存入生成的txt文件：
      示例：../tiny_vid/turtle/000151.JPEG  1  29 38 108 84 2
    其中参数：
      ../tiny_vid/turtle/000151.JPEG：图片路径
      1 ： 表示图像中只有1个物体待检测
      29 38 108 84：分别为 xmin, ymin, xmax, ymax，表示bounding box的位置
      2 ： 物体类别
    """
    classes = {'car': 0, 'bird': 1, 'turtle': 2, 'dog': 3, 'lizard': 4}
    for dirname in classes.keys():
        bbox_dic = {}
        with open(pjoin(defualt_path, dirname+'_gt.txt'), 'r') as f:
            for n, line in enumerate(f.readlines()):
                line = line.strip().split()
                bbox_dic[line[0]] = line[1:]
                if n == 179:  # 每个类别的图像一共有180张图像（0-179）
                    break
        with open(pjoin(defualt_path, 'train_images.txt'), 'a') as f:
            for i in range(1, 151):  # 前150张图像用于training（1-150）
                imgname = '000000'
                pad0 = 6 - len(str(i))
                imgname = imgname[:pad0]+str(i)+'.JPEG'
                imgpath = pjoin(pjoin(defualt_path, dirname), imgname)
                imageclass = str(classes[dirname])
                imgbbox = ' '.join(bbox_dic[str(i)])
                f.write('\t'.join([imgpath, '1', imgbbox, imageclass])+'\n')

        with open(pjoin(defualt_path, 'test_images.txt'), 'a') as f:
            for i in range(151, 181):  # 后30张图像用于testing（151-180）
                imgname = '000000'
                pad0 = 6 - len(str(i))
                imgname = imgname[:pad0]+str(i)+'.JPEG'
                imgpath = pjoin(pjoin(defualt_path, dirname), imgname)
                imageclass = str(classes[dirname])
                imgbbox = ' '.join(bbox_dic[str(i)])
                f.write('\t'.join([imgpath, '1', imgbbox, imageclass])+'\n')


class tiny_vid_loader(data.Dataset):
    """
    功能：
      构造一个用于tiny_vid数据集的迭代器，实现图像增强，包括随机变形、翻转和裁切
    """
    img_size =128

    def __init__(self, defualt_path='./tiny_vid', mode='train', transform='some augmentation'):
        """
        defualt_path: '../tiny_vid'
        mode : 'train' or 'test'
        """
        if not (os.path.exists(pjoin(defualt_path, 'train_images.txt')) and os.path.exists(pjoin(defualt_path, 'test_images.txt'))):
            train_test_txt(defualt_path)
        self.filelist=[]
        self.class_coor = []
        self.mode = True if mode == 'train' else False

        with open(pjoin(defualt_path, mode+'_images.txt')) as f:
            for line in f.readlines():
                line = line.strip().split()
                self.filelist.append(line[0])
                self.class_coor.append([int(i) for i in line[2:]])
        self.ToTensor = transforms.ToTensor()
        # 使用ImageNet的均值和标准差对图像进行正则化，三个分量的顺序为RGB
        self.Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.data_encoder = DataEncoder()
        self.transform = transform

    def random_distort(self, img, brightness_delta=32/255., contrast_delta=0.5, saturation_delta=0.5, hue_delta=0.1):
        '''
        功能：
          对图像进行随机distort，从而实现data augmentation.
        参数:
          img: 输入图像
          brightness_delta: 亮度的偏移量, 范围在[1-delta,1+delta].
          contrast_delta: 对比度的偏移量, 范围在[1-delta,1+delta].
          saturation_delta: 饱和度的偏移量, 范围在[1-delta,1+delta].
          hue_delta: (float) 色相的偏移量, 范围在[-delta,delta].
        返回:
          img: 完成增强的图像
        '''
        def brightness(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(brightness=delta)(img)
            return img

        def contrast(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(contrast=delta)(img)
            return img

        def saturation(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(saturation=delta)(img)
            return img

        def hue(img, delta):
            if random.random() < 0.5:
                img = transforms.ColorJitter(hue=delta)(img)
            return img

        img = brightness(img, brightness_delta)
        if random.random() < 0.5:
            img = contrast(img, contrast_delta)
            img = saturation(img, saturation_delta)
            img = hue(img, hue_delta)
        else:
            img = saturation(img, saturation_delta)
            img = hue(img, hue_delta)
            img = contrast(img, contrast_delta)
        return img

    def random_flip(self, img, boxes):
        '''
        功能：
          随机翻转图像，同时根据翻转的情况调整bbox的位置（只在水平方向翻转）
          如果原始的bbox为(xmin, ymin, xmax, ymax)，则翻转后的bbox为(w-xmax, ymin, w-xmin, ymax).
        参数:
          img: 原始图像
          boxes: (tensor) 物体边框，维度 [#obj, 4].
        返回:
          img: 翻转后的图像
          boxes: 翻转后的bbox
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[2]
            xmax = w - boxes[0]
            boxes[0] = xmin
            boxes[2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''
        功能：
          随机裁剪图像，同时根据裁剪的情况调整bbox的位置
        参数:
          img: 原始图像
          boxes: (tensor) 图像中的物体边框, 维度 [4,].
          labels: (tensor) 物体类别标签, 维度 [1,].
        返回:
          img: 裁剪后的图像
          selected_boxes: (tensor) 裁剪后的边框
          labels: 物体类别标签
        '''
        imw, imh = img.size
        boxes = torch.unsqueeze(boxes,dim=0) # expand [1,4]
        # print(boxes)
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                # random.randrange(min,max)包含min 不包含max
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                img = img.crop((x, y, x+w, y+h))
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                # print(selected_boxes, mask)
                return img, selected_boxes, labels#labels[mask]

    def __getitem__(self, index):
        '''
        功能：
          每次加载图像时，则使用上面定义的方法对图像进行随机augmentation和标准化处理
        '''
        imgpath = self.filelist[index]
        gt_class = np.array(self.class_coor[index][-1], dtype = np.float32)
        gt_bbox = np.array(self.class_coor[index][:-1], dtype = np.float32)
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            gt_class, gt_bbox = torch.Tensor(gt_class), torch.Tensor(gt_bbox)
            if self.mode:
                img = self.random_distort(img)
                img, gt_bbox = self.random_flip(img, gt_bbox)
                img, gt_bbox, gt_class = self.random_crop(img, gt_bbox, gt_class)
                w, h = img.size
                gt_bbox /= torch.Tensor([w, h, w, h]).expand_as(gt_bbox)
                img = transforms.Resize((128, 128))(img)
            img = self.ToTensor(img)
            img = self.Normalize(img)
        else:
            img, gt_class, gt_bbox = self.ToTensor(img), torch.Tensor(gt_class), torch.Tensor(gt_bbox/128.)
            img = self.Normalize(img)
        return img, gt_class.long(), (gt_bbox*128).squeeze()

    def __len__(self):
        return len(self.filelist)
