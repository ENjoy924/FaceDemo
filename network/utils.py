from itertools import product

import math

import numpy as np
import torch


class PriorBox():
    def __init__(self, target_size=640):
        self.target_size = target_size
        self.steps = [8, 16, 32]
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.features = [(math.ceil(self.target_size / s), math.ceil(self.target_size / s)) for s in self.steps]

    def create_box(self):
        priors = []
        for i, f in enumerate(self.features):
            min_sizes = self.min_sizes[i]
            for k, p in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    h = min_size / self.target_size
                    w = min_size / self.target_size
                    cx = [s * self.steps[i] / self.target_size for s in [k + 0.5]]
                    cy = [s * self.steps[i] / self.target_size for s in [p + 0.5]]
                    for x, y in list(zip(cx, cy)):
                        priors.append([x, y, w, h])
        priors = np.array(priors, dtype=np.float32)
        priors = np.reshape(priors, (-1, 4))
        return priors


class CustomLoss():
    def __init__(self):
        self.threshold = 0.2
        self.variance = 0.1

    def __call__(self, priors, pred_result, truth_result):
        pred_box, pred_land, pred_class = pred_result
        batch_size = truth_result.shape[0]
        prior_num = priors.shape[0]
        truth_bbox = torch.zeros(size=(batch_size, prior_num, 4))
        truth_land = torch.zeros(size=(batch_size, prior_num, 10))
        truth_conf = torch.zeros(size=(batch_size, prior_num))
        for idx in range(batch_size):
            self.match(priors, truth_bbox, truth_land, truth_conf, truth_result, idx)



    def point_form(self, boxes):
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], dim=-1)

    def intersect(self, boxa, boxb):
        A = boxa.size(0)  # priors num
        B = boxb.size(0)  # face num
        boxa = boxa.unsqueeze(0).expand(B, A, 4)
        boxb = boxb.unsqueeze(1).expand(B, A, 4)
        min_xy = torch.maximum(boxa[:, :, :2], boxb[:, :, :2])
        max_xy = torch.minimum(boxa[:, :, 2:], boxb[:, :, 2:])
        wh = torch.clip(max_xy - min_xy, 0, 1)
        return wh[:, :, 0] * wh[:, :, 1]

    def jaccard(self, boxa, boxb):
        boxa = self.point_form(boxa)  # priors
        inter = self.intersect(boxa, boxb)
        boxa_area_xy = torch.clip(boxa[:, 2:] - boxa[:, :2], 0, 1)
        boxa_area = boxa_area_xy[:, 0] * boxa_area_xy[:, 1]
        boxb_area_xy = torch.clip(boxb[:, 2:] - boxb[:, :2], 0, 1)
        boxb_area = boxb_area_xy[:, 0] * boxb_area_xy[:, 1]
        boxa_area = boxa_area.unsqueeze(0).expand_as(inter)
        boxb_area = boxb_area.unsqueeze(1).expand_as(inter)

        union = boxa_area + boxb_area - inter
        return inter / union

    def encode_bbox(self, boxes, priors):
        box = torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,
                         boxes[:, 2:] - boxes[:, :2]], dim=-1)
        cxcy = box[:, :2] - priors[:, :2]
        dwdh = box[:, 2:] - priors[:, 2:]
        cxcy /= (cxcy * self.variance)
        dwdh /= box[:, 2:]
        dwdh = torch.log(dwdh) / self.variance
        return torch.cat([cxcy, dwdh], dim=1)

    def encode_land(self, boxes, priors):
        boxes = boxes.view(boxes.size(0), 5, 2)
        cx = priors[:, 0].unsqueeze(1).expand(boxes.shape[0], 5).unsqueeze(2)
        cy = priors[:, 1].unsqueeze(1).expand(boxes.shape[0], 5).unsqueeze(2)
        cw = priors[:, 2].unsqueeze(1).expand(boxes.shape[0], 5).unsqueeze(2)
        ch = priors[:, 3].unsqueeze(1).expand(boxes.shape[0], 5).unsqueeze(2)
        prior = torch.cat([cx, cy, cw, ch], dim=-1)
        dxdy = boxes - prior[:, :, :2]
        dxdy /= (self.variance * prior[:, :, 2:])
        dxdy = dxdy.view(dxdy.size(0), -1)
        return dxdy

    def match(self, priors, truth_bbox, truth_land, truth_conf, truth_result, idx):
        label_bbox = truth_result[:, :4]
        label_land = truth_result[:, 4:14]
        label_conf = truth_result[:, 14]
        overlaps = self.jaccard(priors, label_bbox)

        prior_best_overlap, prior_best_index = overlaps.max(dim=1, keepdim=True)
        valid_best_index = prior_best_overlap > self.threshold
        valid_best_index_filter = prior_best_index[valid_best_index]
        if valid_best_index_filter.shape[0] <= 0:
            truth_conf[idx, :] = 0
            truth_bbox[idx, :] = 0
            return

        truth_best_overlap, truth_best_index = overlaps.max(dim=0, keepdim=True)
        truth_best_overlap.index_fill_(1, prior_best_index[:, 0], 1)
        for j in range(len(prior_best_index[:, 0])):
            truth_best_index[:, prior_best_index[j, 0]] = j
        truth_best_index = truth_best_index.squeeze(0)
        truth_best_overlap = truth_best_overlap.squeeze(0)
        prior_best_index = prior_best_index.squeeze(1)
        prior_best_overlap = prior_best_overlap.squeeze(1)
        valid_best_index = valid_best_index.squeeze(1)

        exp_conf = label_conf[truth_best_index]
        exp_conf[exp_conf < self.threshold] = 0
        truth_conf[idx] = exp_conf

        exp_bbox = label_bbox[truth_best_index]
        exp_bbox = self.encode_bbox(exp_bbox, priors)
        truth_bbox[idx] = exp_bbox

        exp_land = label_land[truth_best_index]
        exp_land = self.encode_land(exp_land, priors)
        truth_land[idx] = exp_land


def test_custom_loss():
    loss_func = CustomLoss()
    pred_conf = torch.rand(size=(16800, 2))
    pred_bbox = torch.rand(size=(16800, 4))
    pred_land = torch.rand(size=(16800, 10))
    pred_result = [pred_bbox, pred_land, pred_conf]
    priors = torch.rand(size=(16800, 4))
    truth_result = torch.rand(size=(8, 15))
    loss = loss_func(priors, pred_result, truth_result)


def test_prior_box():
    prior = PriorBox()
    box = prior.create_box()
    print(box.shape)


if __name__ == '__main__':
    test_custom_loss()
