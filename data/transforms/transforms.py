# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random
from PIL import Image
import cv2
import sys
import collections
import numpy as np

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomCenterErasing(object):
    """ Randomly Select center region in an image and erase its pixels.
    Args:
         probability: The probability that the Center Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.8, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        # half chance to do center erase and half for original random erase!
        if random.uniform(0, 1) >= 0.5:
            for attempt in range(100):
                area = img.size()[1] * img.size()[2]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[2] and h < img.size()[1]:
                    x1 = random.randint(0, img.size()[1] - h)
                    y1 = random.randint(0, img.size()[2] - w)
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    return img
        else:
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(0.1, 0.4) * area
            aspect_ratio = 1  # make it an square
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            x1 = (img.size()[2] - w) // 2
            y1 = (img.size()[1] - h) // 2

            if img.size()[0] == 3:
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
            else:
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
            return img

        return img


class ResizeKeepRatio(object):
    """Resize the input Image to the given size. Image should be PIL.Image or np.array.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, fillcolor=(124, 116, 104)):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.width = size
            self.height = size
        elif isinstance(size, Iterable) and len(size) == 2:
            [self.width, self.height] = size
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return self.resize_keep_aspect_ratio(img, self.width, self.height, color=self.fillcolor)

    def __repr__(self):
        return self.__class__.__name__ + '(size=({0}, {1}) '.format(self.width, self.height)

    @staticmethod
    def resize_keep_aspect_ratio(image, width, height, interpolation=cv2.INTER_LINEAR, color=(127, 127, 127)):
        """
        keep the image aspect ratio and pad to target size
        :return:
        """

        input_as_PIL_image = isinstance(image, Image.Image)

        if input_as_PIL_image:
            image = np.asarray(image)

        h, w, _ = image.shape
        ratio = min(height / h, width / w)
        resize_h, resize_w = int(h * ratio), int(w * ratio)
        resize_image = cv2.resize(image, (resize_w, resize_h), interpolation=interpolation)
        top = round((height - resize_h) / 2 - 0.1)
        bottom = round((height - resize_h) / 2 + 0.1)
        left = round((width - resize_w) / 2 - 0.1)
        right = round((width - resize_w) / 2 + 0.1)
        pad_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # cv2.namedWindow('seesee', 0)
        # cv2.imshow('seesee', pad_image)
        # cv2.waitKey(0)

        assert pad_image.shape[0] == height and pad_image.shape[1] == width, \
            'shape of pad image is wrong %s x %s' % (pad_image.shape[0], pad_image.shape[1])

        return Image.fromarray(np.uint8(pad_image)) if input_as_PIL_image else pad_image
