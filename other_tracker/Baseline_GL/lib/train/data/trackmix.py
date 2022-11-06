import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from timm.data.random_erasing import RandomErasing
import torch.distributed as dist

from PIL import Image
import random
import numpy as np
import math
import cv2
def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

class TrackMix(object):
    """ Randomly selects a rectangle region in an image and erases its pixels with different mixing operation.
    normal: original random erasing;
    soft: mixing ori with random pixel;
    self: mixing ori with other_ori_patch;
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='self',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

    def global_mix(self, img1, img2=None):
        # input shape (h,w,c)
        if random.uniform(0, 1) >= self.probability:
            mix_img = img1
            return mix_img

        lam = np.float32(np.random.beta(self.mixing_coeff[0],self.mixing_coeff[1]))
        if not img1.shape[0] == img2.shape[0] or not img1.shape[1] == img2.shape[1]:
            # resize parameter (w,h)
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation = cv2.INTER_AREA)
        # b=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # a=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL/debug/temp_GL.png", a)
        # cv2.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL/debug/temp_mix.png", b)
        mix_img = lam * img1 + (1 - lam) * img2
        # c = cv2.cvtColor(mix_img.astype(np.uint8),cv2.COLOR_BGR2RGB)
        # cv2.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL/debug/mix_GL.png", c)


        return mix_img

    # local mix
    def transform_image(self, img, mix_img=None):
        # id = np.float32(
        #                 np.random.beta(self.mixing_coeff[0],
        #                                self.mixing_coeff[1]))
        if random.uniform(0, 1) >= self.probability:
            return img
        if not img.shape[0] == mix_img.shape[0] or not img.shape[1] == mix_img.shape[1]:
            # resize parameter (w,h)
            mix_img = cv2.resize(mix_img, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
        # a=cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # b=cv2.cvtColor(mix_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # cv2.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL/debug/img_{}.png".format(id), a)
        # cv2.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL/debug/sameseq_img_{}.png".format(id), b)

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if self.type == 'normal':
                    m = 1.0
                else:
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))
    
                if self.type == 'self':
                    x2 = random.randint(0, img.shape[0] - h)
                    y2 = random.randint(0, img.shape[1] - w)
                    img[x1:x1 + h,
                        y1:y1 + w,:] = (1 - m) * img[x1:x1 + h, y1:y1 +
                                                   w,:] + m * mix_img[x2:x2 + h,
                                                                y2:y2 + w,:]
                # c = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                # cv2.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL/debug/mixed_img_{}.png".format(id), c)
                return img
      
        return img