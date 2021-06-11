import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import data_augment


class WriderFaceDataset(Dataset):
    def __init__(self, annotation_file):
        super(WriderFaceDataset, self).__init__()
        fp = open(annotation_file)
        self.labels = []
        self.imgs = []
        isFirst = True
        label = []
        for line in fp.readlines():
            line = line.strip()
            if line.startswith("#"):
                if isFirst:
                    isFirst = False
                else:
                    lb = label.copy()
                    self.labels.append(lb)
                    label.clear()
                img_path = line.split(" ")[1]
                self.imgs.append(os.path.join(annotation_file.replace('label.txt', 'images'), img_path))
            else:
                lb = [float(i) for i in line.split(" ")]
                label.append(lb)
        self.imgs = self.imgs[:len(self.imgs) - 1]
        fp.close()

    def transform(self, img, label):
        rand_noise = data_augment.RandNoise()
        color_rand = data_augment.ColorAugment()
        scale_padding = data_augment.ScalePadding()
        rand_blur = data_augment.RandBlur()
        img, label = rand_noise(img, label)
        img, label = color_rand(img, label)
        img, label = rand_blur(img, label)
        img, label = scale_padding(img, label)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        annotations = np.zeros(shape=[0, 15])
        img = self.imgs[idx]
        label = self.labels[idx]
        if len(label) == 0:
            return annotations
        for lb in label:
            annotation = np.zeros([1, 15])
            annotation[0][0] = lb[0]
            annotation[0][1] = lb[1]
            annotation[0][2] = lb[2] + lb[0]
            annotation[0][3] = lb[3] + lb[1]
            annotation[0][4] = lb[4]
            annotation[0][5] = lb[5]
            annotation[0][6] = lb[7]
            annotation[0][7] = lb[8]
            annotation[0][8] = lb[10]
            annotation[0][9] = lb[11]
            annotation[0][10] = lb[13]
            annotation[0][11] = lb[14]
            annotation[0][12] = lb[16]
            annotation[0][13] = lb[17]
            if annotation[0][4] == -1:
                annotation[0][14] = -1
            else:
                annotation[0][14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        img = cv2.imread(img)
        img, label = self.transform(img, annotations)
        return torch.from_numpy(img), annotations


def detect_collate(batch):
    target = []
    imgs = []
    for bat in batch:
        for tp in bat:
            if torch.is_tensor(tp):
                imgs.append(tp)
            else:
                tp = torch.from_numpy(tp).float()
                target.append(tp)
    return torch.stack(imgs, dim=0), target
