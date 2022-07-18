import random
import os
from mytest.nips_corruption import *
from PIL import Image
import cv2

class Corrupt_Transform:
    '''
        level: severity level, default 0
        type: corruption function mode, default random 
    '''
    def __init__(self, level=None, type='random'):
        self.corruption_function = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                                glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
                                elastic_transform, pixelate, jpeg_compression, speckle_noise,
                                gaussian_blur, spatter, saturate, rain]
        self.type=type
        if level:
            self.level=level
        else:
            self.level = random.choice(range(1,6))
        
        if self.type=='random':
            self.corrupt_func = random.choice(self.corruption_function)
        else:
            func_name_list=[f.__name__ for f in self.corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            self.corrupt_func = self.corruption_function[corrupt_idx] 
   
    '''
        function name: corrupt_trans
        describe: random corrupt images
        input: imagefile 
        output: PIL.Image
    '''
    def corrupt_trans(self,imagefile, level=None):
        img = Image.open(imagefile)
        if level:
            self.level = level
        # print(self.corrupt_func.__name__)
        # print(self.type)
        # print(self.level)
        c_img = self.corrupt_func(img, self.level)
        return Image.fromarray(np.uint8(c_img))

    # convert PIL.Image to cv2
    def pil2cv2(pil):
        im_np = np.asarray(pil)
        return cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)

