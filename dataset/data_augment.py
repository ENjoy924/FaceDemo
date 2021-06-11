import cv2
import numpy as np


class DataAugment():
    def __init__(self, p=0.5):
        self.p = p

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        pass

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, img: np.ndarray, label: np.ndarray):
        flag = np.random.rand()
        if flag < self.p:
            img, label = self.aug(img, label)
        return img, label


class IdentAugment(DataAugment):
    def __init__(self):
        self.p = 1

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        return img

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        return img, label


class RandNoise(DataAugment):
    def __init__(self):
        super(RandNoise, self).__init__()

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        su = 0
        std = np.random.uniform(0, 15)
        noise = np.random.normal(su, std, size=img.shape)
        img = np.asarray(img, dtype=np.float32)
        img += noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        img = self.aug_img(img)
        return img, label


class RandBlur(DataAugment):
    def __init__(self):
        super(RandBlur, self).__init__()

    @staticmethod
    def gaussian_blur(img):
        kernel = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        return img

    @staticmethod
    def median_blur(img):
        kernel = np.random.choice([3, 5])
        img = cv2.medianBlur(img, kernel)
        return img

    @staticmethod
    def blur(img):
        kernel = np.random.choice([3, 5])
        img = cv2.blur(img, (kernel, kernel))
        return img

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        func = np.random.choice([self.gaussian_blur, self.median_blur, self.blur])
        img = func(img)
        return img

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        img = self.aug_img(img)
        return img, label


class ColorAugment(DataAugment):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        super(ColorAugment, self).__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        rn = np.arange(0, 256)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        h_ltu = (r[0] * rn) % 180
        s_ltu = np.clip(r[1] * rn, 0, 255)
        v_ltu = np.clip(r[2] * rn, 0, 255)
        img = cv2.merge([cv2.LUT(h, h_ltu), cv2.LUT(s, s_ltu), cv2.LUT(v, v_ltu)]).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        img = self.aug_img(img)
        return img, label


class ScaleNoPadding(DataAugment):
    def __init__(self, target_size=640):
        super(ScaleNoPadding, self).__init__()
        self.p = 1
        self.target_size = target_size

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        dh, dw = self.target_size / h, self.target_size / w
        r = min(dh, dw)
        th, tw = round(r * h, r * w)
        img = cv2.resize(img, (th, tw))
        return img, r

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        img, r = self.aug_img(img)
        label = label * r
        return img, label


class ScalePadding(DataAugment):
    def __init__(self, target_size=640):
        super(ScalePadding, self).__init__()
        self.p = 1
        self.target_size = target_size

    def aug_img(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        fh, fw = self.target_size / h, self.target_size / w
        r = min(fh, fw)
        th, tw = round(r * h), round(r * w)
        img = cv2.resize(img, (th, tw))
        dh = self.target_size - th
        dw = self.target_size - tw
        dh = dh / 2
        dw = dw / 2
        top, bottom = round(dw - 0.1), round(dw + 0.1)
        left, right = round(dh - 0.1), round(dh + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[114, 114, 114])

        return img, top, left, r

    def aug(self, img: np.ndarray, label: np.ndarray) -> tuple:
        img, top, left, r = self.aug_img(img)
        label *= r
        label[:, 0:3:2] += left
        label[:, 1:4:2] += top
        label[:, 4:14:2] += left
        label[:, 5:14: 2] += top
        return img, label
